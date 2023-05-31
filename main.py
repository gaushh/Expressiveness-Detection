import streamlit as st
import pandas as pd
import os
from keypoint_extraction import process_keypoints
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px


# Function to fetch feature data from the CSV file
def fetch_feature_data():
    df = pd.read_csv('data.csv')  # Replace 'data.csv' with the path to your CSV file
    return df.values.tolist()

# Function to save the updated feature data to the CSV file
def save_feature_data(feature_data, columns):
    df = pd.DataFrame(feature_data, columns=columns)
    df.to_csv('data.csv', index=False)  # Replace 'data.csv' with the path to your CSV file

# Function to display the feature data table
def display_feature_table(columns):
    feature_data = fetch_feature_data()
    df = pd.DataFrame(feature_data, columns=columns)
    st.subheader("Feature Data")
    st.dataframe(df)

# Function to display the metric chart
def display_metric_chart(metric_name, metric_values):
    print(metric_name, metric_values)
    st.subheader(metric_name)
    df = pd.DataFrame({metric_name: metric_values})
    st.bar_chart(df)

def display_speaker_metrics(curr_speaker_name):
    feature_data = pd.read_csv('data.csv')  # Replace 'data.csv' with the path to your CSV file
    # Get unique speaker names
    speaker_names = feature_data['speaker_name'].unique()

    # get columns except the first one 
    for metric in feature_data.columns[1:]:  # Exclude 'speaker_name'
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
        # fig.add_trace(go.Box(x=list(feature_data[metric]), y=list(feature_data[metric]), name=curr_speaker_name))

        # Adjust layout
        fig.update_layout(xaxis_tickangle=-45)

        # Display the plot in Streamlit
        st.plotly_chart(fig)


import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# def display_box_plots(curr_speaker_name):
#     feature_data = pd.read_csv('data.csv')
#
#     # Create a box plot for each metric
#     for metric in feature_data.columns[1:]:
#         fig = px.box(feature_data, y=metric)
#         st.plotly_chart(fig)

def display_box_plots(curr_speaker_name):
    feature_data = pd.read_csv('data.csv')

    # Filter the data for the current speaker
    curr_speaker_data = feature_data.loc[feature_data['speaker_name'] == curr_speaker_name]

    # Create a box plot for each metric
    for metric in feature_data.columns[1:]:
        fig = px.box(feature_data, y=metric)

        # Highlight the value for the current speaker
        if not curr_speaker_data.empty:
            curr_value = curr_speaker_data[metric].iloc[0]
            fig.add_scatter(y=[curr_value], mode='markers', marker=dict(color='red', size=8), name=curr_speaker_name)

        st.plotly_chart(fig)



# Function to save the uploaded video as a temp file and process the keypoints
def process_uploaded_video(uploaded_file, speaker_name, columns):
    temp_file_path = os.path.join(os.getcwd(), 'temp_video.mp4')
    with open(temp_file_path, 'wb') as temp_file:
        temp_file.write(uploaded_file.read())
    keypoints_data = process_keypoints(temp_file_path)
    os.remove(temp_file_path)  # Delete the temp file

    # Add speaker name to the keypoints data
    keypoints_data['speaker_name'] = speaker_name

    # Append the keypoints data to the feature data
    feature_data = fetch_feature_data()
    row_data = []
    for column in columns:
        row_data.append(keypoints_data[column])
    feature_data.append(row_data)
    # Save the updated feature data to the CSV file
    save_feature_data(feature_data, columns)

    return keypoints_data

# Main application
def main():
    st.title("Expressiveness Dashboard")

    # Video upload and processing
    st.subheader("Upload and Process Video")
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov"])

    curr_speaker_name = st.text_input("Speaker Name")
    columns = ["speaker_name", 
                   "avg_total_movement", 
                   "avg_openness", 
                   "percentage_body_turning", 
                   "percentage_body_turn_front", 
                   "percentage_body_turn_back", 
                   "percentage_smile", 
                   "percentage_neutral"]
    if uploaded_file is not None and curr_speaker_name != "":
        # Process keypoints using FastAPI
        keypoints_data = process_uploaded_video(uploaded_file, curr_speaker_name, columns)

        # Plot metrics against the database metrics
        feature_data = fetch_feature_data()
        
        for metric_name in columns:
            metric_values = [row[1] for row in feature_data if row[-1] == curr_speaker_name]
            metric_values.append(keypoints_data[metric_name])
            # display_metric_chart(metric_name, metric_values)

    # Display the feature data table
    display_speaker_metrics(curr_speaker_name)
    display_box_plots(curr_speaker_name)
    display_feature_table(columns)
    st.subheader("Processed Video")
    video_path = "results/res.mp4"
    # Display the video
    video_file = open(video_path, "rb")
    video_bytes = video_file.read()
    st.video(video_bytes)


if __name__ == "__main__":
    main()
