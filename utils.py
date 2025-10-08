from dataclasses import dataclass, field
import logging
import os
import numpy as np
import pandas as pd
import shap
from typing import Optional, Union
from lightgbm import LGBMClassifier

from feature_engine.encoding import WoEEncoder

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
)


def determine_datatypes(X: pd.DataFrame, features: list):
    """Categorizes features into continuous, categorical, boolean, and all_null types."""
    meta = [i for i in X.columns if i not in features]
    df = X[features].copy()

    # Standardize various string representations of nulls to np.nan to ensure
    # consistent null handling before data type inference.
    null_representations = [None, "None", "none", "nan", "<NA>", "NA", "N/A", "null"]
    df.replace(null_representations, np.nan, inplace=True)

    # Attempt to convert object columns to numeric types. This is crucial for columns
    # that contain numbers stored as strings.
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")

    # Treat datetime columns as categorical features by converting them to strings.
    dt_cols = df.select_dtypes(
        include=["datetime64[ns]", "datetime64[ns, UTC]"]
    ).columns
    df[dt_cols] = df[dt_cols].astype(str)

    # Identify columns that are entirely null, as they may require special handling.
    all_null = [c for c in df.columns if df[c].isnull().all()]

    # Identify boolean-like columns, which are defined as having two or fewer
    # unique values, making them suitable for boolean casting.
    bools = []
    for col in df.select_dtypes(include=["object", "bool"]).columns:
        if col not in all_null and df[col].dropna().nunique() <= 2:
            try:
                # Check if the column can be safely cast to a nullable boolean type.
                df[col].astype("boolean")
                bools.append(col)
            except (ValueError, TypeError):
                continue

    # Categorize the remaining columns as either categorical (object) or
    # continuous (numeric).
    remaining = [c for c in df.columns if c not in all_null + bools]
    cats = df[remaining].select_dtypes(include=["object", "category"]).columns.tolist()
    conts = df[remaining].select_dtypes(include=np.number).columns.tolist()

    return {
        "meta": meta,
        "continuous": conts,
        "categorical": cats,
        "boolean": bools,
        "all_null": all_null,
        "model": conts + cats + bools,
    }


def get_average_shap_values(model: LGBMClassifier, X: pd.DataFrame) -> pd.Series:
    """
    Calculates the mean absolute SHAP values for each feature, providing a
    quantitative measure of feature importance.
    """
    explainer = shap.TreeExplainer(model)

    # For binary classifiers, shap_values returns a list of two arrays (one for each
    # class). We are typically interested in the SHAP values for the positive class.
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_values_positive_class = shap_values[1]
    else:
        # For regression models, only one set of SHAP values is returned.
        shap_values_positive_class = shap_values

    # The mean absolute SHAP value across all samples indicates the average magnitude
    # of each feature's contribution to the model's prediction.
    mean_abs_shap = np.abs(shap_values_positive_class).mean(axis=0)
    shap_series = pd.Series(mean_abs_shap, index=X.columns)
    shap_series = shap_series.sort_values(ascending=False)

    return shap_series


def get_performance_metrics(y_true, y_pred, sample_weight=None):
    """
    Generates a dictionary of performance metrics for a binary classification model.

    Args:
        y_true: True labels.
        y_pred: Predicted probabilities for the positive class.
        sample_weight: Optional sample weights for weighted metrics.

    Returns:
        A dictionary containing key performance metrics.
    """
    metrics = {}

    metrics['positive_label_rate'] = y_true.sum()/y_true.shape[0]
    metrics['auc'] = roc_auc_score(y_true, y_pred)

    if sample_weight is not None:
        metrics['weighted_auc'] = roc_auc_score(y_true, y_pred, sample_weight=sample_weight)

    metrics['average_precision'] = average_precision_score(y_true, y_pred)

    # Standardized AP adjusts for the base rate (positive label rate), providing a
    # more comparable metric across datasets with different class imbalances.
    positive_rate = np.mean(y_true)
    metrics['standardized_ap'] = (metrics['average_precision'] - positive_rate) / (1 - positive_rate) if positive_rate < 1 else np.nan

    # The AUC ratio (also known as the Gini coefficient or Accuracy Ratio) measures
    # the ratio of the area under the actual cumulative gains curve to the ideal one.
    # It provides a summary of model performance across all thresholds.
    y_true_series = pd.Series(y_true).reset_index(drop=True)
    y_pred_series = pd.Series(y_pred)

    sorted_indices = np.argsort(-y_pred_series.values)
    y_true_sorted = y_true_series.iloc[sorted_indices].reset_index(drop=True)
    
    cum_true_positives = y_true_sorted.cumsum()
    total_instances = len(y_true_series)
    total_positives = y_true_series.sum()
    
    actual_auc = cum_true_positives.sum()
    ideal_cum = np.minimum(np.arange(1, total_instances + 1), total_positives)
    ideal_auc = ideal_cum.sum()
    
    auc_ratio = actual_auc / ideal_auc if ideal_auc != 0 else np.nan
    metrics['lift_auc_ratio'] = auc_ratio

    # To find the best operating point for the model, we identify the threshold
    # that maximizes the F1 score, a balance between precision and recall.
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    
    f1_scores = (2 * precision * recall) / (precision + recall + 1e-9)
    
    best_f1_idx = np.argmax(f1_scores[:-1])
    
    metrics['f1_optimized_precision'] = precision[best_f1_idx]
    metrics['f1_optimized_recall'] = recall[best_f1_idx]
    metrics['f1_optimized_f1'] = f1_scores[best_f1_idx]
    metrics['f1_optimized_threshold'] = thresholds[best_f1_idx]

    return metrics


@dataclass
class DataTypeCoercer(BaseEstimator, TransformerMixin):
    """Coerces columns to specific data types based on feature categories."""
    feature_types: dict

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Applies data type coercion to the DataFrame."""
        X_out = X.copy()

        null_representations = [None, "None", "none", "nan", "<NA>", "NA", "N/A", "null"]
        X_out.replace(null_representations, np.nan, inplace=True)

        continuous_cols = self.feature_types.get("continuous", [])
        if continuous_cols:
            X_out[continuous_cols] = X_out[continuous_cols].astype("float64")

        categorical_cols = self.feature_types.get("categorical", [])
        if categorical_cols:
            X_out[categorical_cols] = X_out[categorical_cols].astype("object")

        boolean_cols = self.feature_types.get("boolean", [])
        if boolean_cols:
            # This conversion requires that nulls have already been filled, as integer
            # types cannot accommodate NaN values.
            X_out[boolean_cols] = X_out[boolean_cols].astype("int")

        return X_out
    

@dataclass
class NullFillTransformer(TransformerMixin):
    """Fills null values in a DataFrame based on categorized feature types."""

    feature_types: dict
    continuous_nullfill_offset: int = 1
    categorical_nullfill: str = "unknown"
    boolean_nullfill: float = -1.0
    default_nullfill: int = 0

    def fit(self, X: pd.DataFrame, y=None):
        """Calculate fill values for continuous features based on their minimums."""
        X_clean = X.copy()

        self.continuous_fill_values_ = {}
        continuous_cols = self.feature_types.get("continuous", [])
        if continuous_cols:
            # Coercing to numeric handles cases where numbers are stored as strings.
            # Errors are converted to NaN, which are then handled by the fill logic.
            numeric_cols = X_clean[continuous_cols].apply(pd.to_numeric, errors='coerce')
            self.continuous_fill_values_ = (
                numeric_cols.min() - self.continuous_nullfill_offset
            ).to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply null filling transformations to the DataFrame."""
        X_out = X.copy()

        continuous_cols = self.feature_types.get("continuous", [])
        if continuous_cols and self.continuous_fill_values_:
            X_out[continuous_cols] = X_out[continuous_cols].fillna(
                self.continuous_fill_values_
            )

        cat_cols = self.feature_types.get("categorical", [])
        if cat_cols:
            X_out[cat_cols] = X_out[cat_cols].fillna(self.categorical_nullfill)

        bool_cols = self.feature_types.get("boolean", [])
        if bool_cols:
            # Boolean columns are cast to float to accommodate a numeric fill value.
            X_out[bool_cols] = X_out[bool_cols].astype("float64").fillna(
                self.boolean_nullfill
            )

        all_null_cols = self.feature_types.get("all_null", [])
        if all_null_cols:
            X_out[all_null_cols] = X_out[all_null_cols].fillna(self.default_nullfill)

        return X_out



@dataclass
class OnehotWrapper(TransformerMixin):
    """A wrapper for scikit-learn's OneHotEncoder to integrate with the pipeline."""
    feature_types: dict

    def __post_init__(self):
        self.categorical_features: list = self.feature_types["categorical"]

    def fit(self, X: pd.DataFrame, y=None):
        self.onehot_transform = OneHotEncoder(
            handle_unknown="ignore", drop="if_binary", min_frequency=0.001
        )
        self.onehot_transform.fit(X[self.categorical_features])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        d_onehotted = self.onehot_transform.transform(
            X[self.categorical_features].astype(str)
        )
        d_onehotted = pd.DataFrame(
            d_onehotted.toarray(),
            columns=self.onehot_transform.get_feature_names_out(),
            index=X.index,
        )
        X = X.drop(columns=self.categorical_features)
        d_onehotted = pd.concat([X, d_onehotted], axis=1)
        return d_onehotted


@dataclass
class WoEEncoderWrapper(BaseEstimator, TransformerMixin):
    """
    Wrapper for the feature-engine WoEEncoder to integrate with the project's pipeline.
    """
    feature_types: dict
    fill_value: float = 1e-6

    def __post_init__(self):
        """Initializes the encoder after the dataclass is created."""
        from feature_engine.encoding import WoEEncoder as FeatureEngineWoEEncoder
        
        self.categorical_features: list = self.feature_types.get("categorical", [])
        # The fill_value is used to prevent division by zero errors when a category
        # contains records of only one class.
        self.encoder = FeatureEngineWoEEncoder(
            variables=self.categorical_features, fill_value=self.fill_value
        )

    def fit(self, X: pd.DataFrame, y: pd.Series):
        # feature-engine requires y to be a Series, so we ensure that here.
        if isinstance(y, pd.DataFrame):
            if y.shape[1] == 1:
                y = y.iloc[:, 0]
            else:
                raise ValueError("WoEEncoder expects y to be a Series or single-column DataFrame.")

        self.encoder.fit(X, y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.encoder.transform(X)


@dataclass
class StandardScalerWrapper(TransformerMixin):
    """A wrapper for scikit-learn's StandardScaler to integrate with the pipeline."""
    feature_types: dict

    def __post_init__(self):
        self.continuous: list = self.feature_types["continuous"]

    def fit(self, X: pd.DataFrame, y=None):
        self.scaler_transform = StandardScaler()
        self.scaler_transform.fit(X[self.continuous])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        d_continuous = X[self.continuous]
        d = self.scaler_transform.transform(d_continuous)
        d = pd.DataFrame(d, columns=d_continuous.columns, index=d_continuous.index)
        X = X.drop(columns=self.continuous)
        d_standard = pd.concat([X, d], axis=1)
        return d_standard


@dataclass
class LightGBMWrapper(BaseEstimator):
    """
    A wrapper for the LightGBM classifier, optionally including hyperparameter
    tuning via GridSearchCV.
    """
    hyper_parameters: dict = None
    folds: int = 3
    njobs: int = 8
    seed: int = 1005

    def fit(self, X: pd.DataFrame, y: Union[list, pd.Series, np.array], sample_weight=None) -> pd.DataFrame:
        # If hyperparameters are provided, perform a grid search to find the best model.
        if self.hyper_parameters:
            self.model = LGBMClassifier(
                objective="binary",
                random_state=self.seed,
            )
            self.grid_search = GridSearchCV(
                estimator=self.model,
                param_grid=self.hyper_parameters,
                scoring="roc_auc",
                n_jobs=self.njobs,
                cv=KFold(n_splits=self.folds),
                verbose=2,
            )
            self.grid_search.fit(X, y, sample_weight=sample_weight)
            self.model_ = self.grid_search.best_estimator_
        else:
            # Otherwise, fit a standard LightGBM model.
            self.model_ = LGBMClassifier(
                objective="binary",
                random_state=self.seed,
            )
            self.model_.fit(X, y, sample_weight=sample_weight)

        return self

    def predict(self, X):
        return self.model_.predict_proba(X)[:, 1]