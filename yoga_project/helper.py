import cv2
import mediapipe as mp
from tqdm import tqdm
import pickle
import numpy as np

def process_video(input_path, output_path, frame_rate):
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    # Using Model 0 (Lite) for maximum speed
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=0)
    
    cap = cv2.VideoCapture(input_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    
    skip_step = max(1, int(original_fps // frame_rate))  # Skip frames to achieve target FPS
    
    # 320x180 is very low res but works for Pose AI
    width, height = 320, 180 
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, frame_rate, (width, height))

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

    with open(f"data/pkl/{filename}.pkl", 'wb') as f:
        pickle.dump(pose_data, f)

    cap.release()
    out.release()

    return pose_data

def standardize(poses):
    # use hips as origin, normalize by arm length
    ret = []
    for pose in poses:
        hip_center = [(pose[11][0] + pose[12][0]) / 2, (pose[11][1] + pose[12][1]) / 2, (pose[11][2] + pose[12][2]) / 2]
        arm_length = ((pose[15][0] - pose[11][0])**2 + (pose[15][1] - pose[11][1])**2 + (pose[15][2] - pose[11][2])**2) ** 0.5
        new_lms = []
        for lm in pose:
            new_lms.append([(lm[0] - hip_center[0]) / arm_length, (lm[1] - hip_center[1]) / arm_length, (lm[2] - hip_center[2]) / arm_length])
        ret.append(new_lms)
    return ret

def diff(poses):
    diff = []
    for i, pose1 in enumerate(poses[:-1]):

        pose2 = poses[i + 1]
        p1 = np.array(pose1)
        p2 = np.array(pose2)
        
        # Define the bilateral pairs based on your 23-point list:
        # 0: Nose (no pair)
        # 1-2: Shoulders, 3-4: Elbows, 5-6: Wrists
        # 13-14: Hips, 15-16: Knees, 17-18: Ankles, etc.
        pairs = [(1,2), (3,4), (5,6), (13,14), (15,16), (17,18), (19,20), (21,22)]
        
        # 1. Start with the Nose (Index 0)
        total_diff = np.linalg.norm(p1[0] - p2[0])
        
        # 2. Iterate through body pairs
        for i, j in pairs:
            # Direct Match: L1->L2 and R1->R2
            direct = np.linalg.norm(p1[i] - p2[i]) + np.linalg.norm(p1[j] - p2[j])
            
            # Swapped Match: L1->R2 and R1->L2
            swapped = np.linalg.norm(p1[i] - p2[j]) + np.linalg.norm(p1[j] - p2[i])
            
            # Take the minimum of the two to account for flips
            total_diff += min(direct, swapped)
            
        diff.append(total_diff)
    return diff