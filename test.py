import cv2
import os
import numpy as np

def max_pooling_video(input_video_path, output_video_path, pooling_size=2):
    # Open the video file
    cap = cv2.VideoCapture(input_video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {input_video_path}")
        return
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Prepare the output video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps , (frame_width, frame_height))

    # Initialize a list to store frames for pooling
    frame_buffer = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_buffer.append(frame)
        
        if len(frame_buffer) == pooling_size:
            # Perform max pooling on the buffered frames
            max_pooled_frame = np.mean(frame_buffer, axis=0)
            out.write(max_pooled_frame.astype(np.uint8))
            frame_buffer.pop(0)

    # Release resources
    cap.release()
    out.release()
    print(f"Max pooled video saved at {output_video_path}")
pooling_size = 3
input_list = ['01_Bird', '02_City', '03_Forest']
for name in input_list:
    input_path = f"C:/Users/dudgb2380/Downloads/naturalistic_video/direction/{name}.mp4_combined.avi"
    output_folder = "C:/Users/dudgb2380/Downloads/naturalistic_video/direction/"
    
    max_pooling_video(input_path, output_folder+os.path.basename(input_path).split('.')[0]+f"_pooling_{pooling_size}.avi", pooling_size)
