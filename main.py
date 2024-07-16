# %%

# 필요한 라이브러리 임포트
import numpy as np
import os
import mediapipe as mp
import cv2
from my_functions import *  # 사용자 정의 함수 임포트
import keyboard
from tensorflow.keras.models import load_model

# 데이터 디렉터리 경로 설정
PATH = os.path.join('data')

# 데이터 디렉터리 내의 파일 목록을 통해 행동 레이블 배열 생성
actions = np.array(os.listdir(PATH))

# 학습된 모델 로드
model = load_model('my_model.keras')

# 리스트 초기화
keypoints = []
predictions = []
last_prediction = ""

# 카메라 접근 및 성공적으로 열렸는지 확인
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot access camera.")
    exit()

# 전체적인 객체를 생성하여 사인 예측을 수행
with mp.solutions.holistic.Holistic(min_detection_confidence=0.75, min_tracking_confidence=0.75) as holistic:
    while cap.isOpened():
        # 카메라에서 프레임 읽기
        _, image = cap.read()
        
        # 이미지 처리 및 my_functions.py의 image_process 함수를 사용하여 사인 랜드마크 획득
        results = image_process(image, holistic)
        
        # my_functions.py의 draw_landmarks 함수를 사용하여 이미지에 사인 랜드마크 그리기
        draw_landmarks(image, results)
        
        # my_functions.py의 keypoint_extraction 함수를 사용하여 포즈 랜드마크에서 키포인트 추출
        keypoints.append(keypoint_extraction(results))

        # 30 프레임이 누적되었는지 확인
        if len(keypoints) == 30:
            # 키포인트 리스트를 넘파이 배열로 변환
            keypoints = np.array(keypoints)
            
            # 로드된 모델을 사용하여 키포인트에 대한 예측 수행
            prediction = model.predict(keypoints[np.newaxis, :, :])
            
            # 예측 결과를 predictions 리스트에 추가
            predictions.append(prediction[0])  # Ensure the prediction is added as a 1D array
            
            # 다음 프레임 세트를 위해 키포인트 리스트 초기화
            keypoints = []

            # 누적된 예측 결과의 평균 계산
            avg_prediction = np.mean(predictions, axis=0)
            predicted_action_index = np.argmax(avg_prediction)
            predicted_action = actions[predicted_action_index]
            avg_confidence = avg_prediction[predicted_action_index]

            # 다음 프레임 세트를 위해 predictions 리스트 초기화
            predictions = []

            # 평균 신뢰도가 임계값을 초과하는지 확인
            if avg_confidence > 0.9:
                last_prediction = predicted_action
            else:
                last_prediction = "Fail"

        # 이미지에 예측 결과 표시
        cv2.putText(image, f'Prediction: {last_prediction}', 
                    (20, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if last_prediction != "Fail" else (0, 0, 255), 2, cv2.LINE_AA)

        # 디스플레이에 이미지 표시
        cv2.imshow('Camera', image)
        
        cv2.waitKey(1)

        # 'q' 키가 눌렸는지 확인하여 루프 종료
        if keyboard.is_pressed('q'):
            break

        # 'Camera' 창이 닫혔는지 확인하여 루프 종료
        if cv2.getWindowProperty('Camera', cv2.WND_PROP_VISIBLE) < 1:
            break

    # 카메라 해제 및 모든 창 닫기
    cap.release()
    cv2.destroyAllWindows()

