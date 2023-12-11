# Copyright 2023 The MediaPipe Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Main scripts to run pose landmarker."""

import argparse
import sys
import time
import json

import cv2
import mediapipe as mp
import numpy as np

import time, libcamera
from picamera2 import Picamera2, Preview

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Global variables to calculate FPS
COUNTER, FPS = 0, 0
START_TIME = time.time()
DETECTION_RESULT = None

def tuloste(name1, name2, index, lista):
  print('{{'.format())
  print('   {0}: {{'.format(name1))
  print('          {0}: {{'.format(name2))
  print(f'                 x: {lista[index].x}')
  print(f'                 y: {lista[index].y}')
  print('                }}'.format())
  print('          }}'.format())
  print(' }}'.format())

def run(model: str, num_poses: int,
        min_pose_detection_confidence: float,
        min_pose_presence_confidence: float, min_tracking_confidence: float,
        output_segmentation_masks: bool,
        camera_id: int, width: int, height: int) -> None:
    """Continuously run inference on images acquired from the camera.

  Args:
      model: Name of the pose landmarker model bundle.
      num_poses: Max number of poses that can be detected by the landmarker.
      min_pose_detection_confidence: The minimum confidence score for pose
        detection to be considered successful.
      min_pose_presence_confidence: The minimum confidence score of pose
        presence score in the pose landmark detection.
      min_tracking_confidence: The minimum confidence score for the pose
        tracking to be considered successful.
      output_segmentation_masks: Choose whether to visualize the segmentation
        mask or not.
      camera_id: The camera id to be passed to OpenCV.
      width: The width of the frame captured from the camera.
      height: The height of the frame captured from the camera.
  """

    # Start capturing video input from the camera
    #cap = cv2.VideoCapture(camera_id)
    #cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    picam = Picamera2()
    config = picam.create_preview_configuration(main={'format': 'BGR888', "size": (640, 480)})
    config["transform"] = libcamera.Transform(hflip=0, vflip=1)
    picam.configure(config)
    picam.start()

    # Visualization parameters
    row_size = 50  # pixels
    left_margin = 24  # pixels
    text_color = (0, 0, 0)  # black
    font_size = 1
    font_thickness = 1
    fps_avg_frame_count = 10
    overlay_alpha = 0.5
    mask_color = (100, 100, 0)  # cyan

    def save_result(result: vision.PoseLandmarkerResult,
                    unused_output_image: mp.Image, timestamp_ms: int):
        global FPS, COUNTER, START_TIME, DETECTION_RESULT

        # Calculate the FPS
        if COUNTER % fps_avg_frame_count == 0:
            FPS = fps_avg_frame_count / (time.time() - START_TIME)
            START_TIME = time.time()

        DETECTION_RESULT = result
        COUNTER += 1

    # Initialize the pose landmarker model
    base_options = python.BaseOptions(model_asset_path=model)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_poses=num_poses,
        min_pose_detection_confidence=min_pose_detection_confidence,
        min_pose_presence_confidence=min_pose_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
        output_segmentation_masks=output_segmentation_masks,
        result_callback=save_result)
    detector = vision.PoseLandmarker.create_from_options(options)

    # Continuously capture images from the camera and run inference
    while True:
        image = picam.capture_array()
        #if not success:
        #    sys.exit(
        #        'ERROR: Unable to read from webcam. Please verify your webcam settings.'
        #    )#
        
        #image = cv2.flip(image, 1)

        # Convert the image from BGR to RGB as required by the TFLite model.
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        # Run pose landmarker using the model.
        detector.detect_async(mp_image, time.time_ns() // 1_000_000)

        # Show the FPS
        fps_text = 'FPS = {:.1f}'.format(FPS)
        text_location = (left_margin, row_size)
        current_frame = image
        cv2.putText(current_frame, fps_text, text_location,
                    cv2.FONT_HERSHEY_DUPLEX,
                    font_size, text_color, font_thickness, cv2.LINE_AA)

        if DETECTION_RESULT:
            # Draw landmarks.
            print("*********************")
            #https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
            try:
                tuloste("upper_body", "nose", 0,DETECTION_RESULT.pose_landmarks.landmark._values)
                tuloste("upper_body", "left_eye_inner", 0,DETECTION_RESULT.pose_landmarks.landmark._values)
                tuloste("upper_body", "left_eye", 0,DETECTION_RESULT.pose_landmarks.landmark._values)
                tuloste("upper_body", "left_eye_outer", 0,DETECTION_RESULT.pose_landmarks.landmark._values)
                tuloste("upper_body", "right_eye_inner", 0,DETECTION_RESULT.pose_landmarks.landmark._values)
                tuloste("upper_body", "right_eye", 0,DETECTION_RESULT.pose_landmarks.landmark._values)
                tuloste("upper_body", "right_eye_outer", 0,DETECTION_RESULT.pose_landmarks.landmark._values)
                tuloste("upper_body", "left_ear", 0,DETECTION_RESULT.pose_landmarks.landmark._values)
                tuloste("upper_body", "right_ear", 0,DETECTION_RESULT.pose_landmarks.landmark._values)
                tuloste("upper_body", "mouth_left", 0,DETECTION_RESULT.pose_landmarks.landmark._values)
                tuloste("upper_body", "mouth_right", 0,DETECTION_RESULT.pose_landmarks.landmark._values)
                tuloste("upper_body_joint", "left_shoulder", 0,DETECTION_RESULT.pose_landmarks.landmark._values)
                tuloste("upper_body_joint", "right_shoulder", 0,DETECTION_RESULT.pose_landmarks.landmark._values)
                tuloste("upper_body_joint", "left_elbow", 0,DETECTION_RESULT.pose_landmarks.landmark._values)
                tuloste("upper_body_joint", "right_elbow", 0,DETECTION_RESULT.pose_landmarks.landmark._values)
                tuloste("upper_bod_jointy", "left_wrist", 0,DETECTION_RESULT.pose_landmarks.landmark._values)
                tuloste("upper_body_joint", "right_wrist", 0,DETECTION_RESULT.pose_landmarks.landmark._values)
                tuloste("hands", "left_pinky", 0,DETECTION_RESULT.pose_landmarks.landmark._values)
                tuloste("hands", "right_pinky", 0,DETECTION_RESULT.pose_landmarks.landmark._values)
                tuloste("hands", "left_index", 0,DETECTION_RESULT.pose_landmarks.landmark._values)
                tuloste("hands", "right_index", 0,DETECTION_RESULT.pose_landmarks.landmark._values)
                tuloste("hands", "left_thumb", 0,DETECTION_RESULT.pose_landmarks.landmark._values)
                tuloste("hands", "right_thumb", 0,DETECTION_RESULT.pose_landmarks.landmark._values)
                tuloste("lower_body", "left_hip", 0,DETECTION_RESULT.pose_landmarks.landmark._values)
                tuloste("lower_body", "right_hip", 0,DETECTION_RESULT.pose_landmarks.landmark._values)
                tuloste("lower_body", "left_knee", 0,DETECTION_RESULT.pose_landmarks.landmark._values)
                tuloste("lower_body", "right_knee", 0,DETECTION_RESULT.pose_landmarks.landmark._values)
                tuloste("lower_body", "left_ankle", 0,DETECTION_RESULT.pose_landmarks.landmark._values)
                tuloste("lower_body", "right_ankle", 0,DETECTION_RESULT.pose_landmarks.landmark._values)
                tuloste("feet", "left_heel", 0,DETECTION_RESULT.pose_landmarks.landmark._values)
                tuloste("feet", "right_heel", 0,DETECTION_RESULT.pose_landmarks.landmark._values)
                tuloste("feet", "left_foot_index", 0,DETECTION_RESULT.pose_landmarks.landmark._values)
                tuloste("feet", "right_foot_index", 0,DETECTION_RESULT.pose_landmarks.landmark._values)
            except Exception as error :
                print(f"ERR {error}")
            print(DETECTION_RESULT.pose_landmarks.landmark._values())
            print("*********************")
            
            for pose_landmarks in DETECTION_RESULT.pose_landmarks:
                # Draw the pose landmarks.
                pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
               
                
                pose_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y,
                                                    z=landmark.z) for landmark
                    in pose_landmarks
                ])
                mp_drawing.draw_landmarks(
                    current_frame,
                    pose_landmarks_proto,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing_styles.get_default_pose_landmarks_style())

        if (output_segmentation_masks and DETECTION_RESULT):
            if DETECTION_RESULT.segmentation_masks is not None:
                segmentation_mask = DETECTION_RESULT.segmentation_masks[0].numpy_view()
                mask_image = np.zeros(image.shape, dtype=np.uint8)
                mask_image[:] = mask_color
                condition = np.stack((segmentation_mask,) * 3, axis=-1) > 0.1
                visualized_mask = np.where(condition, mask_image, current_frame)
                current_frame = cv2.addWeighted(current_frame, overlay_alpha,
                                                visualized_mask, overlay_alpha,
                                                0)

        cv2.imshow('pose_landmarker', current_frame)

        dic = {}
        for mark, data_point in zip(mp_pose.PoseLandmark, DETECTION_RESULT.pose_landmarks):
            dic[mark.value] = dict(landmark = mark.name, 
                x = data_point.x,
                y = data_point.y,
                z = data_point.z,
                visibility = data_point.visibility)
            
        json_object = json.dumps(dic, indent=2)

        # Stop the program if the ESC key is pressed.
        if cv2.waitKey(1) == 27:
            break

    detector.close()
    #cap.release()
    cv2.destroyAllWindows()
    picam.stop()

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--model',
        help='Name of the pose landmarker model bundle.',
        required=False,
        default='pose_landmarker.task')
    parser.add_argument(
        '--numPoses',
        help='Max number of poses that can be detected by the landmarker.',
        required=False,
        default=1)
    parser.add_argument(
        '--minPoseDetectionConfidence',
        help='The minimum confidence score for pose detection to be considered '
             'successful.',
        required=False,
        default=0.5)
    parser.add_argument(
        '--minPosePresenceConfidence',
        help='The minimum confidence score of pose presence score in the pose '
             'landmark detection.',
        required=False,
        default=0.5)
    parser.add_argument(
        '--minTrackingConfidence',
        help='The minimum confidence score for the pose tracking to be '
             'considered successful.',
        required=False,
        default=0.5)
    parser.add_argument(
        '--outputSegmentationMasks',
        help='Set this if you would also like to visualize the segmentation '
             'mask.',
        required=False,
        action='store_true')
    # Finding the camera ID can be very reliant on platform-dependent methods.
    # One common approach is to use the fact that camera IDs are usually indexed sequentially by the OS, starting from 0.
    # Here, we use OpenCV and create a VideoCapture object for each potential ID with 'cap = cv2.VideoCapture(i)'.
    # If 'cap' is None or not 'cap.isOpened()', it indicates the camera ID is not available.
    parser.add_argument(
        '--cameraId', help='Id of camera.', required=False, default=0)
    parser.add_argument(
        '--frameWidth',
        help='Width of frame to capture from camera.',
        required=False,
        default=1280)
    parser.add_argument(
        '--frameHeight',
        help='Height of frame to capture from camera.',
        required=False,
        default=960)
    args = parser.parse_args()

    run(args.model, int(args.numPoses), args.minPoseDetectionConfidence,
        args.minPosePresenceConfidence, args.minTrackingConfidence,
        args.outputSegmentationMasks,
        int(args.cameraId), args.frameWidth, args.frameHeight)


if __name__ == '__main__':
    main()
