import pandas as pd

# Define the data for the dummy CSV file
data = {
    "total_movement": [10, 20, 15],
    "avg_openness": [0.8, 0.6, 0.9],
    "time_leaning_forward": [5, 2, 7],
    "time_leaning_backward": [3, 4, 1],
    "time_head_right": [2, 6, 3],
    "time_head_left": [4, 3, 2],
    "time_head_up": [6, 1, 4],
    "time_head_down": [1, 5, 2],
    "speaker_name": ["Test1", "Test2", "Test3"]
}

# Create a DataFrame from the data
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.to_csv('data.csv', index=False)