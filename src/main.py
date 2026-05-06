import mediapipe as mp

# 初始化
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

# 打开摄像头
cap = cv2.VideoCapture(0)

# 姿态检测
pose_detector = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 转 RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose_detector.process(frame_rgb)

    # 画骨架
    if result.pose_landmarks:
        mp_draw.draw_landmarks(
            frame,
            result.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

    cv2.imshow("Pose", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
