import pandas as pd
data = pd.read_csv('sample_data_snippet.csv')

print(data.columns)
asset_ID = data["asset_id"]
print(asset_ID)
start_time = data["start_date_time"]
print(start_time)

data[data["asset_id"] == 11942011].to_csv('Alby_Will')
