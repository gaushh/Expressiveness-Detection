# Expressiveness-Detection

Expressiveness Detection is a project that aims to analyze and measure the expressiveness and body movements of individuals in videos. It utilizes computer vision techniques and the MediaPipe library to extract keypoints from video frames and calculate various metrics related to expressiveness.

## Features

- Keypoint Extraction: The project uses the MediaPipe library to extract keypoints from video frames. These keypoints represent the positions of different body parts and facial landmarks.

- Pose Openness: The project calculates the openness of a pose by analyzing the positions of specific body keypoints. This metric provides insights into the openness and confidence of an individual's body posture.

- Leaning Direction: The project determines the leaning direction (forward or backward) based on the positions of the nose and hips in the pose. This metric helps understand the body's inclination and engagement.

- Head Direction: The project tracks the movement of the head and determines its horizontal and vertical direction (left, right, up, or down). This metric provides information about the individual's head movement patterns.

- Metric Visualization: The project visualizes the calculated metrics using charts and tables. This allows for a comprehensive analysis of expressiveness and body movements over time.

## Requirements

- Python 3.7 or higher
- OpenCV
- Mediapipe
- NumPy
- SciPy

## Installation

1. Clone the repository:

   ```shell
   git clone https://github.com/your-username/expressiveness-detection.git

2. Navigate to the project directory:

	```shell
	cd expressiveness-detection

3. Install the required dependencies:

	```shell
	pip install -r requirements.txt

4. Usage
	- Prepare the video data:
		- Place the video files you want to analyze in a directory.
		- Update the path to the video directory in the main.py file.

	- Run the application:
		```shell
		python main.py

	- Upload a video file:
		- Click on the "Upload a video file" button in the Streamlit application.
		- Select a video file (maximum duration of 10 seconds) for analysis.
		- Enter the name of the speaker.

	- View the results:
		- The application will process the video and calculate various metrics.
		- The feature data table and metric charts will be displayed in the Streamlit application.

5. Contributing
	Contributions are welcome! If you have any ideas, suggestions, or improvements, please open an issue or submit a pull request.

6. License
	This project is licensed under the MIT License.

