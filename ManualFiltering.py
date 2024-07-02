import torch
import torch.nn as nn
import numpy as np
import cv2
import os
import manual_filters

def get_file_name_from_path(path):
    file_name = os.path.basename(path)
    return file_name[:file_name.rfind('.')]
    

class Simple3DCNN(nn.Module):
    def __init__(self, direction_filters):
        super(Simple3DCNN, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_pool = nn.MaxPool3d(kernel_size=(2, 1, 1))

    def forward(self, x):
        return self.max_pool(x)
    
def frame_diff_filtering(video_path, output_file):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    time = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height), isColor=False)
    
    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        return
    
    prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    frame_idx = 0
    
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_diff = cv2.absdiff(gray_frame, prev_frame)
        prev_frame = gray_frame
        
        out.write(frame_diff)
        frame_idx += 1
        print("diff filtering...  %.2f%% (%d / %d) done"%(frame_idx / time * 100, frame_idx, time), end='\r')
    cap.release()
    out.release()
    print(f"Frame difference video saved to {output_file}")

def apply_threshold_to_video(input_path, output_path, threshold):
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=False)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary_frame = cv2.threshold(gray_frame, threshold, 255, cv2.THRESH_BINARY)
        out.write(binary_frame)
    
    cap.release()
    out.release()

def process_videos_in_folder(input_folder, threshold):
    output_folder = os.path.join(input_folder, f'thresholding_{threshold}')
    os.makedirs(output_folder, exist_ok=True)
    
    threshold *= 255
    for filename in os.listdir(input_folder):
        if filename.endswith('.mp4'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            
            apply_threshold_to_video(input_path, output_path, threshold)
            print(f"Processed {filename} and saved to {output_path}")



def movement_filtering(source_file, destination_folder, filters, filter_depth):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Simple3DCNN(filters).to(device)
    cap = cv2.VideoCapture(source_file)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    time = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    filter_dir = ['origin','up','down','right','left']
    batch_size = 50

    output_h, output_w = h // 4, w // 4

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_videos = [
        cv2.VideoWriter(f'{destination_folder}/{get_file_name_from_path(source_file)}_{filter_dir[i]}.mp4', fourcc, fps, (output_w, output_h), isColor=False)
        for i in range(5)
    ]

# 0-padding을 위한 초기 프레임 설정
    padding_frame = np.zeros((h, w), dtype=np.float32)
    frames = [padding_frame for _ in range(filter_depth)]

    frame_idx = 0
    batch_frames = []
    
    while frame_idx < time:
        ret, frame = cap.read()
        if not ret:
            break
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        batch_frames.append(gray_frame)
        
        if len(batch_frames) == batch_size:
            frames.extend(batch_frames)
            if len(frames) > filter_depth + batch_size - 1:
                frames = frames[-(filter_depth + batch_size - 1):]
            
            frames_tensor = torch.tensor(np.array(frames), dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, depth+batch-1, h, w)
            with torch.no_grad():
                output = model(frames_tensor)  # (1, 5, depth+batch-1, h, w)
            
            for j in range(batch_size):
                for i in range(output.shape[1]):
                    frame_out = output[0, i, filter_depth + j - 1].cpu().numpy()
                    frame_out = (((frame_out - frame_out.min()) / (frame_out.max() - frame_out.min())) * 255).astype(np.uint8)
                    out_videos[i].write(cv2.merge([frame_out]))
            
            batch_frames = []
            print("filtering...  %.2f%% (%d / %d) done" % ((frame_idx + 1) / time * 100, frame_idx + 1, time), end='\r')
        
        frame_idx += 1
    
    # Process remaining frames in batch
    if batch_frames:
        frames.extend(batch_frames)
        if len(frames) > filter_depth + len(batch_frames) - 1:
            frames = frames[-(filter_depth + len(batch_frames) - 1):]
        
        frames_tensor = torch.tensor(np.array(frames), dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(frames_tensor)
        
        for j in range(len(batch_frames)):
            for i in range(output.shape[1]):
                frame_out = output[0, i, filter_depth + j - 1].cpu().numpy()
                frame_out = (((frame_out - frame_out.min()) / (frame_out.max() - frame_out.min())) * 255).astype(np.uint8)
                out_videos[i].write(cv2.merge([frame_out]))

    cap.release()
    for out in out_videos:
        out.release()

    print("영상 저장 완료")
    

def filtering(filters, filter_depth, destination_folder_name):
    #desktop_path = "/host/c/Users/전영규/Desktop/filtering_results"
    desktop_path = "../filtering_results"
    folder_path = "./naturalistic_video"
    bird_video_path = f"{folder_path}/01_Bird.mp4"
    city_video_path = f"{folder_path}/02_City.mp4"
    forest_video_path = f"{folder_path}/03_Forest.mp4"
    name_list = [
        os.path.splitext(bird_video_path)[0]+"_grayscale_video.mp4",
        os.path.splitext(city_video_path)[0]+"_grayscale_video.mp4",
        os.path.splitext(forest_video_path)[0]+"_grayscale_video.mp4"]
    
    
    destination_folder = f"{desktop_path}/{destination_folder_name}"
    print(destination_folder_name)
    for source_file in name_list:
        print(source_file)
        #source_file = f"{folder_path}/03_Forest_grayscale_video.mp4"
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)
        # x = torch.cat([origin, x], dim=1)
        # diff_video_path = os.path.splitext(source_file)[0]+"_diff_video.mp4"
        # if not os.path.exists(diff_video_path):
        #     frame_diff_filtering(source_file, diff_video_path)
        
        #movement_filtering(diff_video_path, destination_folder, filters, filter_depth)
        print()
        movement_filtering(source_file, destination_folder, filters, filter_depth)
        print()
    for threshold in np.arange(0.5, 0.6, 0.02).tolist():
        print(f"{threshold} thresholding... " )
        process_videos_in_folder(destination_folder, threshold)
    

if __name__ == "__main__":
    # filters, filter_depth = manual_filters.standard_2by2()
    # filtering(filters, filter_depth,  "diff_standard_2by2")
    
    filters, filter_depth = manual_filters.standard_3by3()
    filtering(filters, filter_depth,  "standard_3by3_no_diff")
    # filters, filter_depth = manual_filters.standard_2by3()
    # filtering(filters, filter_depth,  "standard_2by3")
    
    # filters, filter_depth = manual_filters.line_3by3()
    # filtering(filters, filter_depth,  "line_3by3_no_diff")
    
    # filters, filter_depth = manual_filters.laplacian_5by5()
    # filtering(filters, filter_depth, "lap_5by5")
    
    # filters, filter_depth = manual_filters.diff_5by1()
    # filtering(filters, filter_depth,"diff_5by1")
    
    # for threshold in np.arange(0.5, 1.0, 0.1).tolist():
    #     print(f"{threshold} thresholding... " )
    #     process_videos_in_folder("/host/c/Users/hy105/Desktop/filtering_results/lap_5by5", threshold)
    
    
    
    

    