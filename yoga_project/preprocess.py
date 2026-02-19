import cv2
import mediapipe as mp
from tqdm import tqdm
import pickle
def process_video(input_path, output_video):
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    # Using Model 0 (Lite) for maximum speed
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=0)
    
    cap = cv2.VideoCapture(input_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    
    skip_step = 3
    target_fps = original_fps / skip_step
    
    # 320x180 is very low res but works for Pose AI
    width, height = 320, 180 
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, target_fps, (width, height))

    filename = input_path.split('/')[-1].split('.')[0]
    pose_data = []

    with tqdm(total=total_frames, desc=f"Processing {filename}") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 1. Downsample for speed
            resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_NEAREST)
            img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            results = pose.process(img_rgb)
            
            if results.pose_landmarks:
                # 1. Hide unwanted head landmarks (1-10)
                # This ensures they don't get drawn on the video
                for i in range(1, 11):
                    results.pose_landmarks.landmark[i].visibility = 0
                
                # 2. Extract Data: Nose (0) + Shoulders down (11-32)
                head_and_body = [results.pose_landmarks.landmark[0]] + list(results.pose_landmarks.landmark[11:])
                landmarks = [[lm.x, lm.y, lm.z] for lm in head_and_body]
                pose_data.append(landmarks)
                
                # 3. Draw on the frame
                # Connections between nose and shoulders will still draw 
                # because points 0, 11, and 12 are still visible.
                mp_drawing.draw_landmarks(
                    resized, 
                    results.pose_landmarks, 
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 200, 0), thickness=1)
                )

            # 3. Save frame with drawings
            out.write(resized)
            
            # 4. Jump ahead
            for _ in range(skip_step - 1):
                cap.grab()
            
            pbar.update(skip_step)

    cap.release()
    out.release()
    with open(f"data/pkl/{filename}.pkl", 'wb') as f:
        pickle.dump(pose_data, f)
    return pose_data