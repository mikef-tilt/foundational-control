import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt

def calculate_growth_stats(input_file, histogram_output, summary_output):
    # Load the data
    # input_file = '/Users/mike.fang/projects/analysis/foundational-control/cache/first_loans.parquet'
    df = pd.read_parquet(input_file)

    # Convert 'loan_predicted_at' to datetime objects
    df['loan_predicted_at'] = pd.to_datetime(df['loan_predicted_at'])

    # Set 'loan_predicted_at' as the index
    df.set_index('loan_predicted_at', inplace=True)

    # Resample by week and count the number of loans
    weekly_loans = df.resample('W').size()
    
    # Week-over-week growth
    weekly_growth = weekly_loans.pct_change().fillna(0) * 100
    
    # trim top and bottom 2% to remove outliers
    lower_bound = weekly_growth.quantile(0.02)
    upper_bound = weekly_growth.quantile(0.98)
    weekly_growth = weekly_growth[(weekly_growth > lower_bound) & (weekly_growth < upper_bound)]

    # Replace inf values resulting from division by zero with 0
    weekly_growth.replace([np.inf, -np.inf], 0, inplace=True)
    avg_weekly_growth = weekly_growth.mean()
    median_weekly_growth = weekly_growth.median()
    std_weekly_growth = weekly_growth.std()
    min_weekly_growth = weekly_growth.min()
    max_weekly_growth = weekly_growth.max()

    # Group values > 100 into the 100 bin for visualization
    visual_growth = weekly_growth.copy()
    visual_growth[visual_growth > 100] = 100

    # Create and save the histogram
    plt.figure(figsize=(12, 7))
    # Define clear bins, ensuring the last bin catches everything >= 100
    bins = np.arange(-100, 101, 10)
    bins = np.append(bins, visual_growth.max()) # Ensure the last bin includes the max value
    plt.hist(visual_growth, bins=bins, edgecolor='black', alpha=0.7)
    
    # Add vertical lines for mean and median (of the original data)
    plt.axvline(avg_weekly_growth, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {avg_weekly_growth:.2f}%')
    
    plt.title('Distribution of Weekly Loan Growth (Outliers > 100% grouped)')
    plt.xlabel('Weekly Growth (%)')
    plt.ylabel('Frequency')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.savefig(histogram_output)
    print(f"Histogram saved to {histogram_output}")

    # Output summary statistics to a text file
    summary_stats = (
        f"Loan Growth Summary Statistics:\n"
        f"--------------------------------\n"
        f"Average Weekly Growth: {avg_weekly_growth:.2f}%\n"
        f"Standard Deviation of Weekly Growth: {std_weekly_growth:.2f}%\n"
        f"Minimum Weekly Growth: {min_weekly_growth:.2f}%\n"
        f"Maximum Weekly Growth: {max_weekly_growth:.2f}%\n"
    )
    with open(summary_output, 'w') as f:
        f.write(summary_stats)
    print(f"Summary stats saved to {summary_output}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate first loan growth statistics.')
    parser.add_argument('--input', default='cache/first_loans.parquet', help='Input parquet file path.')
    parser.add_argument('--histogram-output', default='plots/weekly_growth_histogram.png', help='Output histogram file path.')
    parser.add_argument('--summary-output', default='loan_growth_summary.txt', help='Output summary statistics file path.')
    args = parser.parse_args()
    calculate_growth_stats(args.input, args.histogram_output, args.summary_output)
