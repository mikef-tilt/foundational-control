import argparse
import json
import pandas as pd
import numpy as np
import os
import pandas as pd
import pickle as pk
from typing import Union, Optional, Tuple
from dataclasses import dataclass, field
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, StandardScaler
import shap
from xgboost import XGBClassifier
from utils import *
from transformers import *

import pandas as pd
import numpy as np
import pandas as pd
from typing import Union
from dataclasses import dataclass, field
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    FunctionTransformer,
    StandardScaler,
    PolynomialFeatures,
)
from xgboost import XGBClassifier



def get_inputs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, help="training or inference")
    parser.add_argument("--config_file", type=str, help="params for execution")
    parser.add_argument("--raw_data_file", type=str, help="raw data input file")
    parser.add_argument("--output_dir", type=str, help="directory for outputs")
    args = parser.parse_args()
    return args

@dataclass
class ColumnSlicer(TransformerMixin):
    feature_types: dict

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.feature_types["model"]]


@dataclass
class RawDataProcessor(TransformerMixin):
    feature_types: dict

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        pipe = Pipeline(
            [
                (
                    "fixing_column_names",
                    FunctionTransformer(self.fix_categorical_classes),
                ),
            ]
        )
        return pipe.transform(X)

    def fix_categorical_classes(self, X: pd.DataFrame) -> pd.DataFrame:
        for i in self.feature_types["categorical"]:
            X[i] = (
                X[i]
                .str.replace(r"[^A-Za-z0-9]+", "_", regex=True)
                .str.strip("_")
                .str.replace(r"_+", "_", regex=True)
            )
        return X


@dataclass
class PolynomialFeatureTransformer(TransformerMixin):
    degree: int = 2
    include_bias: bool = True
    interaction_only: bool = False

    def fit(self, X, y=None):
        self.poly_transformer = PolynomialFeatures(
            degree=self.degree,
            include_bias=self.include_bias,
            interaction_only=self.interaction_only,
        )
        self.poly_transformer.fit(X)
        return self

    def transform(self, X):
        X_poly = self.poly_transformer.transform(X)
        feature_names = self.poly_transformer.get_feature_names_out(X.columns)
        return pd.DataFrame(X_poly, columns=feature_names, index=X.index)


@dataclass
class NullFillTransformer(TransformerMixin):
    feature_types: dict
    continuous_nullfill_offset: int = 1
    categorical_nullfill: str = "unknown"
    boolean_nullfill: int = -1.0
    default_nullfill: int = 0

    def fit(self, X: pd.DataFrame, y=None):
        X = X[self.feature_types["continuous"]]
        floored = np.floor(X.min()).astype("Int64")
        sign = floored.apply(lambda x: -1 if x <= 1 else 1)
        place = (
            floored.apply(np.format_float_scientific)
            .astype(str)
            .apply(lambda x: int(x[-2:]))
        )
        continuous_nullfill = sign * (
            10
            ** (place * sign).apply(
                lambda x: np.abs(x - self.continuous_nullfill_offset) if x != 0 else x
            )
        )
        self.continuous_nullfill = continuous_nullfill.to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        for k, v in self.continuous_nullfill.items():
            X[k] = X[k].fillna(v)
        for i in ["<NA>", "nan"]:
            X[self.feature_types["categorical"]] = X[
                self.feature_types["categorical"]
            ].replace({i: np.nan})
        X[self.feature_types["categorical"]] = X[
            self.feature_types["categorical"]
        ].fillna(self.categorical_nullfill)
        X[self.feature_types["boolean"]] = (
            X[self.feature_types["boolean"]]
            .astype("float64")
            .fillna(self.boolean_nullfill)
        )
        X[self.feature_types["all_null"]] = X[self.feature_types["all_null"]].fillna(
            self.default_nullfill
        )
        return X


@dataclass
class OnehotWrapper(TransformerMixin):
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
class StandardScalerWrapper(TransformerMixin):
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
class XGBoostWrapper(BaseEstimator):
    hyper_parameters: dict
    folds: int
    njobs: int = 8
    seed: int = 1005
    monotonic_constraints: dict = None

    def fit(self, X: pd.DataFrame, y: Union[list, pd.Series, np.array]) -> pd.DataFrame:
        if self.monotonic_constraints is None:
            self.model = XGBClassifier(
                objective="binary:logistic",
                tree_method="hist",
                seed=self.seed,
            )
        else:
            self.model = XGBClassifier(
                objective="binary:logistic",
                tree_method="hist",
                seed=self.seed,
                monotone_constraints=self.monotonic_constraints,
            )

        self.grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=self.hyper_parameters,
            scoring="roc_auc",
            n_jobs=self.njobs,
            cv=KFold(n_splits=self.folds),
            verbose=2,
        )
        self.grid_search.fit(X, y)
        self.calibrated_model = CalibratedClassifierCV(
            self.grid_search.best_estimator_,
            method="isotonic",
            cv=self.folds,
            n_jobs=self.njobs,
        )
        self.calibrated_model.fit(X, y.astype(int))
        self.grid_search.best_estimator_.get_booster().feature_names = (
            X.columns.to_list()
        )
        return self

    def predict(self, X):
        return {
            "uncalibrated": self.grid_search.best_estimator_.predict_proba(X)[:, 1],
            "calibrated": self.calibrated_model.predict_proba(X)[:, 1],
        }


@dataclass
class MonoXGBoostWrapper(BaseEstimator):
    monotonic_constraints: dict
    hyper_parameters: dict
    folds: int
    njobs: int = 8
    seed: int = 1005

    def fit(self, X: pd.DataFrame, y: Union[list, pd.Series, np.array]) -> pd.DataFrame:
        self.model = XGBClassifier(
            objective="binary:logistic",
            tree_method="hist",
            seed=self.seed,
            monotone_constraints=self.monotonic_constraints,
        )
        self.grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=self.hyper_parameters,
            scoring="roc_auc",
            n_jobs=self.njobs,
            cv=KFold(n_splits=self.folds),
            verbose=2,
        )
        self.grid_search.fit(X, y)
        self.calibrated_model = CalibratedClassifierCV(
            self.grid_search.best_estimator_,
            method="isotonic",
            cv=self.folds,
            n_jobs=self.njobs,
        )
        self.calibrated_model.fit(X, y.astype(int))
        self.grid_search.best_estimator_.get_booster().feature_names = (
            X.columns.to_list()
        )
        return self

    def predict(self, X):
        return {
            "uncalibrated": self.grid_search.best_estimator_.predict_proba(X)[:, 1],
            "calibrated": self.calibrated_model.predict_proba(X)[:, 1],
        }



def run_base_model(data, config):

    # determining datatypes
    feature_types = determine_datatypes(
        data, features=config["features"]["model_features"]
    )

    # in-time sampling (timestamps not given)
    d_train, d_test, train_meta, test_meta = data_sampler(data, feature_types)

    # initialize pipeline steps
    column_slicer = ColumnSlicer(feature_types=feature_types)
    rawdata_processor = RawDataProcessor(feature_types=feature_types)
    nullfill_transformer = NullFillTransformer(feature_types=feature_types)
    onehot_encoder = OnehotWrapper(feature_types=feature_types)
    standard_scaler = StandardScalerWrapper(feature_types=feature_types)
    model = XGBoostWrapper(
        hyper_parameters=config["modeling"]["hpo"],
        folds=config["modeling"]["cross_validation"],
    )

    # build pipeline
    model_pipeline = Pipeline(
        [
            ("column_slicer", column_slicer),
            ("rawdata_processor", rawdata_processor),
            ("nullfill_transformer", nullfill_transformer),
            ("onehot_encoder", onehot_encoder),
            ("standard_scaler", standard_scaler),
            ("model", model),
        ]
    )

    # fit pipeline
    model_pipeline.fit(d_train, train_meta.y)

    # get processed data
    d_processed_train = model_pipeline[:-1].transform(d_train)
    d_processed_test = model_pipeline[:-1].transform(d_test)

    # perform predictions
    prob_train = pd.DataFrame(model_pipeline[-1].predict(d_processed_train))
    prob_test = pd.DataFrame(model_pipeline[-1].predict(d_processed_test))

    # append results to meta data
    train_meta = pd.concat([train_meta, prob_train], axis=1)
    test_meta = pd.concat([test_meta, prob_test], axis=1)

    # generate tree explainer
    explainer = shap.TreeExplainer(model_pipeline[-1].grid_search.best_estimator_)

    # generate shap matrix
    d_shap_test = pd.DataFrame(
        explainer.shap_values(d_processed_test), columns=d_processed_test.columns
    )

    # caching data
    d_processed_train.to_parquet(base_dir("train.parquet"), index=False)
    d_processed_test.to_parquet(base_dir("test.parquet"), index=False)
    d_shap_test.to_parquet(base_dir("test_shap.parquet"), index=False)
    train_meta.to_parquet(base_dir("train_meta.parquet"), index=False)
    test_meta.to_parquet(base_dir("test_meta.parquet"), index=False)

    # caching model artifacts
    with open(base_dir("pipeline.pkl"), "wb") as f:
        pk.dump(model_pipeline, f)
    with open(base_dir("uncalibrated_model.pkl"), "wb") as f:
        pk.dump(model_pipeline[-1].grid_search.best_estimator_, f)
    with open(base_dir("calibrated_model.pkl"), "wb") as f:
        pk.dump(model_pipeline[-1].calibrated_model, f)
    with open(base_dir("explainer.pkl"), "wb") as f:
        pk.dump(explainer, f)
