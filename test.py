import pandas as pd

feature_data = pd.read_csv('data.csv')
print(list(feature_data["avg_openness"]))