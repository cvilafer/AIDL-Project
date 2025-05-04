import cv2
import mediapipe as mp

from utils.logger import Logger
from models.mediapipe_holistic_landmarks import MediapipeHolisticLandmarks
from models.mediapipe_landmark import MediapipeLandmark

class MediapipeHolistic:
    def __init__(self):
        """
        Initialize the Mediapipe Holistic model.

        Attributes:
            __mp_holistic (mediapipe.python.solutions.holistic.Holistic): Mediapipe Holistic module.
            __mp_drawing (mediapipe.python.solutions.drawing_utils.DrawingSpec): Drawing utility for visualizing landmarks.
        """
        self.__logger = Logger("MediapipeHolistic")
        self.__mp_holistic = mp.solutions.holistic
        self.__mp_drawing = mp.solutions.drawing_utils

    def process_video(self, video_path: str) -> "MediapipeHolisticLandmarks":
        """
        Process a video to extract landmarks using Mediapipe Holistic.

        Args:
            video_path (str): Path to the video file.

        Returns:
            list: List of landmarks for each frame in the video.
        """        
        # Initialize the MediaPipe Holistic model
        with self.__mp_holistic.Holistic(static_image_mode=False, model_complexity=2) as holistic:
            # Open the video file
            video_capture = cv2.VideoCapture(video_path)
            self.__logger.debug(f"Processing video {video_path}...")

            frames_landmarks = []

            while video_capture.isOpened():
                # Extract the frame from the video
                has_next_frame, frame = video_capture.read()

                # Check if the frame was successfully captured
                if not has_next_frame:
                    break

                # Convert the BGR image to RGB
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Process the image and get the landmarks
                results = holistic.process(image_rgb)

                # Convert the landmarks to a dictionary format
                # Append to the list of landmarks for the video
                single_frame_landmarks = MediapipeHolisticLandmarks(
                    face=MediapipeLandmark.to_dict(results.face_landmarks),
                    pose=MediapipeLandmark.to_dict(results.pose_landmarks),
                    left_hand=MediapipeLandmark.to_dict(results.left_hand_landmarks),
                    right_hand=MediapipeLandmark.to_dict(results.right_hand_landmarks)
                )

                # Append frame landmarks to the list of landmarks for the video
                frames_landmarks.append(single_frame_landmarks)

                # Draw the landmarks on the frame
                self.__mp_drawing.draw_landmarks(frame, results.face_landmarks, self.__mp_holistic.FACEMESH_TESSELATION)
                self.__mp_drawing.draw_landmarks(frame, results.pose_landmarks, self.__mp_holistic.POSE_CONNECTIONS)
                self.__mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, self.__mp_holistic.HAND_CONNECTIONS)
                self.__mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, self.__mp_holistic.HAND_CONNECTIONS)

                # Display the frame with landmarks
                cv2.imshow("Mediapipe Holistic", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            # Release the video capture object and close all OpenCV windows
            video_capture.release()
            cv2.destroyAllWindows()
            self.__logger.debug(f"Finished processing video {video_path}.")

        # Return the list of landmarks for the video
        return frames_landmarks