import argparse
import pandas as pd
import numpy as np
import json




def main(input_file, config_file, iterations):
    
    # Load the data
    input_file = '/home/azureuser/localfiles/analyses/foundational-control/cache/first_loans.parquet'
    df = pd.read_parquet(input_file)

    # Load config (not used in this example, but could be for model parameters)
    with open(config_file, 'r') as f:
        config = json.load(f)

    # initialize parameters    
    meta = config['meta']
    features = config['features']
    sample_counts = [len(features)] + config['sample_counts']

    # iterating through samples and n iterations
    for i in sample_counts:
        for j in range(iterations):
            sample = df.sample(n=i, replace=True, random_state=j)
            X = sample[features]
            y = sample[meta['target']]
            
            # Simple model: calculate mean of target
            mean_target = y.mean()
            print(f'Sample size: {i}, Iteration: {j}, Mean {meta["target"]}: {mean_target}')





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate first loan growth statistics.')
    parser.add_argument('--input', default='cache/first_loans.parquet', help='Input parquet file path.')
    parser.add_argument('--config', default='cache/config.json', help='Input config file path.')
    parser.add_argument('--iterations', default=100, help='number of models to build.')
    
    args = parser.parse_args()
    main(args.input, args.iterations)

