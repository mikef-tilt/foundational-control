import pandas as pd
import matplotlib.pyplot as plt
import argparse

def main(input_file, output_file):
    # Load the data
    df = pd.read_parquet(input_file)

    # Convert 'loan_predicted_at' to datetime objects
    df['loan_predicted_at'] = pd.to_datetime(df['loan_predicted_at'])

    # Group by day and count the number of loans
    daily_loans = df.groupby(df['loan_predicted_at'].dt.date).size()

    # Group by week and count the number of loans
    weekly_loans = df.groupby(pd.Grouper(key='loan_predicted_at', freq='W')).size()

    # Create the plot
    plt.figure(figsize=(12, 6))
    # plt.plot(daily_loans.index, daily_loans.values, label='Daily')
    plt.plot(weekly_loans.index, weekly_loans.values, label='Weekly')
    plt.plot(weekly_loans.index, weekly_loans.values * 0.05, label='Weekly 5%', linestyle='--')
    plt.title('Number of Loans')
    plt.xlabel('Date')
    plt.ylabel('Number of Loans')
    plt.grid(True)
    plt.legend()
    
    # Save the plot to the output file
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a plot of daily and weekly loan counts.')
    parser.add_argument('--input', default='cache/first_loans.parquet', help='Input parquet file path.')
    parser.add_argument('--output', default='daily_loans.png', help='Output plot file path.')
    args = parser.parse_args()
    main(args.input, args.output)