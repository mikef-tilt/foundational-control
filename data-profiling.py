import pandas as pd
from ydata_profiling import ProfileReport


data_file_path = '/home/azureuser/localfiles/data/raw_data_ensenada.parquet'
df = pd.read_parquet(data_file_path)

# Down sample the data to 10k data points
df_sample = df.sample(n=10000, random_state=42)

# Generate the profile report
profile = ProfileReport(df_sample, title="Pandas Profiling Report for foundational-control")
profile.to_file("profiling_report.html")