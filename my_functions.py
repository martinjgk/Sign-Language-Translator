import mediapipe as mp
import cv2
import numpy as np

def draw_landmarks(image, results): #랜드마크를 그려주는 애
    """
    Draw the landmarks on the image.

    Args:
        image (numpy.ndarray): The input image.
        results: The landmarks detected by Mediapipe.

    Returns:
        None
    """
    # Set the image back to writable mode before drawing
    image.flags.writeable = True

    # Draw landmarks for left hand
    if results.left_hand_landmarks:#왼손 랜드마크를 찾으면 랜드마크를 그린다.
        mp.solutions.drawing_utils.draw_landmarks(
            image, results.left_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS)

    # Draw landmarks for right hand
    if results.right_hand_landmarks:#오른손 랜드마크를 찾으면 랜드마크를 그린다.
        mp.solutions.drawing_utils.draw_landmarks(
            image, results.right_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS)

    # Draw landmarks for pose (body)
    if results.pose_landmarks:#몸의 랜드마크를 찾으면 랜드마크를 그린다.
        mp.solutions.drawing_utils.draw_landmarks(
            image, results.pose_landmarks, mp.solutions.holistic.POSE_CONNECTIONS)

    # Draw landmarks for face with minimal points (eyes, nose, mouth)
    if results.face_landmarks:#얼굴의 랜드마크를 찾으면 랜드마크를 그린다.
        face_landmarks = results.face_landmarks.landmark
        # Define key points for eyes, nose, and mouth
        key_points = [#key point 특정 랜드마크만 그린다.
            33, 133, 145, 153, 362, 263, 373, 380,  # Left eye
            7, 163, 144, 153, 362, 263, 373, 380,  # Right eye
            1, 4, 5, 45, 275, 281,  # Nose
            61, 291, 78, 308, 80, 82, 312, 14, 13  # Mouth
        ]
        for idx in key_points:
            x = int(face_landmarks[idx].x * image.shape[1])
            y = int(face_landmarks[idx].y * image.shape[0])
            cv2.circle(image, (x, y), 2, (0, 0, 255), -1)

def image_process(image, model):#랜드마크를 추출하는 함수
    """
    Process the image and obtain sign landmarks.

    Args:
        image (numpy.ndarray): The input image.
        model: The Mediapipe holistic object.

    Returns:
        results: The processed results containing sign landmarks.
    """
    # Set the image to read-only mode #앍가 전용으로 바꾸어서 랜드마크를 추출하려고 한다.
    image.flags.writeable = False
    # Convert the image from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Process the image using the model
    results = model.process(image)
    # Convert the image back from RGB to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return results

def keypoint_extraction(results):#랜드마크를 배열에다가 담는 함수.
    """
    Extract the keypoints from the sign landmarks.

    Args:
        results: The processed results containing sign landmarks.

    Returns:
        keypoints (numpy.ndarray): The extracted keypoints.
    """
    # Extract the keypoints for the left hand if present, otherwise set to zeros
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(63)
    # Extract the keypoints for the right hand if present, otherwise set to zeros
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(63)
    
    # Define key points for eyes, nose, and mouth
    face_key_points = [
        33, 133, 145, 153, 362, 263, 373, 380,  # Left eye
        7, 163, 144, 153, 362, 263, 373, 380,  # Right eye
        1, 4, 5, 45, 275, 281,  # Nose
        61, 291, 78, 308, 80, 82, 312, 14, 13  # Mouth
    ]
    
    # Extract the keypoints for the face if present, otherwise set to zeros
    if results.face_landmarks:
        face = np.array([[results.face_landmarks.landmark[idx].x, results.face_landmarks.landmark[idx].y, results.face_landmarks.landmark[idx].z] for idx in face_key_points]).flatten()
    else:
        face = np.zeros(len(face_key_points) * 3)
    
    # Concatenate the keypoints for both hands and the face
    keypoints = np.concatenate([lh, rh, face])
    return keypoints
