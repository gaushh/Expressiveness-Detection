import streamlit as st
import pandas as pd
import os
from keypoint_extraction import process_keypoints

# Function to fetch feature data from the CSV file
def fetch_feature_data():
    df = pd.read_csv('data.csv')  # Replace 'data.csv' with the path to your CSV file
    return df.values.tolist()

# Function to save the updated feature data to the CSV file
def save_feature_data(feature_data):
    df = pd.DataFrame(feature_data, columns=['total_movement', 'avg_openness', 'time_leaning_forward', 'time_leaning_backward', 'time_head_right', 'time_head_left', 'time_head_up', 'time_head_down', 'speaker_name'])
    df.to_csv('data.csv', index=False)  # Replace 'data.csv' with the path to your CSV file

# Function to display the feature data table
def display_feature_table():
    feature_data = fetch_feature_data()
    df = pd.DataFrame(feature_data, columns=['total_movement', 'avg_openness', 'time_leaning_forward', 'time_leaning_backward', 'time_head_right', 'time_head_left', 'time_head_up', 'time_head_down', 'speaker_name'])
    st.subheader("Feature Data")
    st.dataframe(df)

# Function to display the metric chart
def display_metric_chart(metric_name, metric_values):
    print(metric_name, metric_values)
    print("_____________")
    st.subheader(metric_name)
    df = pd.DataFrame({metric_name: metric_values})
    st.bar_chart(df)


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Function to fetch feature data from the CSV file

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

def display_speaker_metrics():
    feature_data = pd.read_csv('data.csv')  # Replace 'data.csv' with the path to your CSV file
    # Get unique speaker names
    speaker_names = feature_data['speaker_name'].unique()

    for metric in feature_data.columns[:-1]:  # Exclude 'speaker_name'
        fig = go.Figure()
        fig.update_layout(
            xaxis=dict(title='Speaker Name'),
            yaxis=dict(title=metric.capitalize()),
            title=f'{metric.capitalize()} Comparison',
        )

        # Plot bar chart for each speaker
        for speaker_name in speaker_names:
            speaker_metrics = feature_data[feature_data['speaker_name'] == speaker_name][metric]
            fig.add_trace(go.Bar(x=[speaker_name], y=speaker_metrics))

        # Adjust layout
        fig.update_layout(xaxis_tickangle=-45)

        # Display the plot in Streamlit
        st.plotly_chart(fig)


# Function to save the uploaded video as a temp file and process the keypoints
def process_uploaded_video(uploaded_file, speaker_name):
    temp_file_path = os.path.join(os.getcwd(), 'temp_video.mp4')
    with open(temp_file_path, 'wb') as temp_file:
        temp_file.write(uploaded_file.read())
    keypoints_data = process_keypoints(temp_file_path)
    os.remove(temp_file_path)  # Delete the temp file

    # Add speaker name to the keypoints data
    keypoints_data['speaker_name'] = speaker_name

    # Append the keypoints data to the feature data
    feature_data = fetch_feature_data()
    feature_data.append([keypoints_data['total_movement'], keypoints_data['avg_openness'],
                         keypoints_data['time_leaning_forward'], keypoints_data['time_leaning_backward'],
                         keypoints_data['time_head_right'], keypoints_data['time_head_left'],
                         keypoints_data['time_head_up'], keypoints_data['time_head_down'], speaker_name])

    # Save the updated feature data to the CSV file
    save_feature_data(feature_data)

    return keypoints_data

# Main application
def main():
    st.title("Expressiveness Dashboard")

    # Video upload and processing
    st.subheader("Upload and Process Video")
    uploaded_file = st.file_uploader("Upload a video file (max 10 seconds)", type=["mp4", "mov"])

    speaker_name = st.text_input("Speaker Name")

    if uploaded_file is not None and speaker_name != "":
        # Process keypoints using FastAPI
        keypoints_data = process_uploaded_video(uploaded_file, speaker_name)

        # Plot metrics against the database metrics
        feature_data = fetch_feature_data()
        metric_names = ['total_movement', 'avg_openness', 'time_leaning_forward', 'time_leaning_backward', 'time_head_right', 'time_head_left', 'time_head_up', 'time_head_down', 'speaker_name']

        for metric_name in metric_names:
            metric_values = [row[1] for row in feature_data if row[-1] == speaker_name]
            metric_values.append(keypoints_data[metric_name])
            # display_metric_chart(metric_name, metric_values)

    # Display the feature data table
    display_feature_table()
    display_speaker_metrics()

if __name__ == "__main__":
    main()
