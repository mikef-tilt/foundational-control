import argparse
import pandas as pd
import numpy as np
import json

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from utils import *


def fit_model_pipeline(X_train, y_train, X_weights, features):
    feature_types = determine_datatypes(X_train, features)

    datatype_tranformer = DataTypeCoercer(feature_types=feature_types)
    nullfill_transformer = NullFillTransformer(feature_types=feature_types)
    woe_encoder = WoEEncoderWrapper(feature_types=feature_types)
    standard_scaler = StandardScalerWrapper(feature_types=feature_types)
    model = LightGBMWrapper()

    # build pipeline
    model_pipeline = Pipeline(
        [
            ("datatype_tranformer", datatype_tranformer),
            ("nullfill_transformer", nullfill_transformer),
            ("woe_encoder", woe_encoder),
            ("standard_scaler", standard_scaler),
            ("model", model),
        ]
    )

    model_pipeline.fit(X_train, y_train, model__sample_weight=X_weights)
    return model_pipeline



def main(input_file, config_file):
    
    # Load the data
    input_file = 'cache/first_loans.parquet' # for testing
    df = pd.read_parquet(input_file)

    # Load config (not used in this example, but could be for model parameters)
    config_file = 'cache/config.json'
    with open(config_file, 'r') as f:
        config = json.load(f)

    # Ensure loan_amount is numeric and clean
    df['loan_amount'] = pd.to_numeric(df['loan_amount'], errors='coerce')
    df.dropna(subset=['loan_amount'], inplace=True)

    # initialize parameters
    seed = 1337
    meta = config['meta_features']
    features = config['features']
    iterations = config['iterations']
    feature_counts = [len(features)] + config['feature_counts']
    sample_counts = config['sample_counts']

    # iterating through samples and n iterations
    for i in sample_counts:
        for j in feature_counts:
            for k in range(iterations):

                i = 100000  # for testing
                j = 151  # for testing

                df_sample = df.sample(n=i, random_state=seed)
                
                if feature_counts != len(features):
                    features_sample = df_average_shaps.head(j).index
                else:
                    features_sample = features

                X = df_sample[features_sample]
                y = df_sample[['loan_is_default_21d']].astype(int)

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
                X_train_weights = df_sample.loc[X_train.index, 'loan_amount']
                X_test_weights = df_sample.loc[X_test.index, 'loan_amount']

                model_pipeline = fit_model_pipeline(
                    X_train,
                    y_train,
                    X_train_weights,
                    features_sample
                    )
                
                y_pred = model_pipeline.predict(X_test)
                performance = get_performance_metrics(
                    y_test.loan_is_default_21d.values,
                    y_pred,
                    X_test_weights.values
                    )
                
                if feature_counts != len(features):
                    df_average_shaps = get_average_shap_values(
                        model_pipeline[-1].model_,
                        model_pipeline[:-1].transform(X_train)
                        )



                




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate first loan growth statistics.')
    parser.add_argument('--input', default='cache/first_loans.parquet', help='Input parquet file path.')
    parser.add_argument('--config', default='cache/config.json', help='Input config file path.')
    parser.add_argument('--iterations', default=100, help='number of models to build.')
    
    args = parser.parse_args()
    main(args.input, args.config)

