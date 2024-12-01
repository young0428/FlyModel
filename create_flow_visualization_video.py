import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from FlowNet import *
from trainer_func import *
from LoadDataset import *
import os

def create_flow_visualization_video(model_name, model_class, video_type=2, config={}):
    """Flow prediction 결과를 시각화하는 동영상 생성 함수"""
    # 비디오 데이터 로드
    video_data, wba_data, total_frame = direction_pred_training_data_preparing_seq(
        "./naturalistic", "experimental_data.mat", 5.625)
    
    # 모델 로드
    base_path = f"./model/{model_name}"
    config_string = ''.join([f"_{key}_{value}" for key, value in config.items()])
    model_path = f"{base_path}/piece_size_1{config_string}/fold_1"
    frame_per_window = get_frame_per_window(model_path)
    
    # FlowNet 모델 초기화 및 로드
    flownet_model = flownet3d([[64, 2], [128, 2], [256, 2]])
    model = model_class(flownet_model, feature_dim=128, 
                       input_size=(frame_per_window, 64, 128, 1))
    model.eval()
    trainer = Trainer(model, loss_function_mse, 1e-4)
    trainer.load(f"{model_path}/best_model.ckpt")
    
    # 출력 비디오 설정
    output_folder = "./flow_visualization_video"
    os.makedirs(output_folder, exist_ok=True)
    video_name = {0: 'bird', 1: 'city', 2: 'forest'}
    output_filename = f"{output_folder}/{video_name[video_type]}_{model_name}{config_string}_flow.avi"
    
    # 원본 비디오의 크기
    h, w = video_data.shape[2:4]
    frame_size = (w, h * 2)  # 상하로 두 영상을 배치
    fps = 15  # 원본 fps의 절반
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_filename, fourcc, fps, frame_size, isColor=True)
    
    try:
        # 2프레임씩 건너뛰면서 처리 (flow prediction은 절반 프레임)
        for frame in range(frame_per_window, total_frame-3, 2):
            if frame % 100 == 0:
                print(f"Processing frame {frame}/{total_frame-3}")
            
            # 입력 데이터 준비
            input_data = torch.tensor(
                video_data[video_type:video_type+1, frame-frame_per_window:frame, :, :, 0:1],
                dtype=torch.float32
            ).to(trainer.device)
            
            # Flow prediction
            with torch.no_grad():
                pred = trainer.model.flownet3d(input_data)
                pred = pred.cpu().numpy()[0]  # (2, H/2, W/2)
            
            # Flow ��측 결과 합치기 (좌우 방향 flow의 합)
            flow_sum = (pred[0] + pred[1])  # (H/2, W/2)
            
            # 원본 크기로 리사이즈
            flow_visualization = cv2.resize(flow_sum, (w, h))
            
            # Target flow 계산 (원본 비디오의 채널 3, 4 합)
            target_flow = video_data[video_type, frame, :, :, 3] + video_data[video_type, frame, :, :, 4]
            
            # 시각화를 위한 정규화
            flow_visualization = (flow_visualization - flow_visualization.min()) / (flow_visualization.max() - flow_visualization.min()) * 255
            target_flow = (target_flow - target_flow.min()) / (target_flow.max() - target_flow.min()) * 255
            
            # 컬러맵 적용
            flow_visualization = cv2.applyColorMap(flow_visualization.astype(np.uint8), cv2.COLORMAP_JET)
            target_flow = cv2.applyColorMap(target_flow.astype(np.uint8), cv2.COLORMAP_JET)
            
            # 상하로 합치기
            combined_frame = np.vstack([flow_visualization, target_flow])
            
            # 프레임 저장
            out.write(combined_frame)
        
        print(f"Successfully saved video: {output_filename}")
        
    except Exception as e:
        print(f"Failed to save video: {str(e)}")
        raise
    
    finally:
        out.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    video_type = 2  # forest
    model_name = "forest_wba_value_full_res_avg_8frames"
    config = {
        "fix": False,
        "pretrained": True
    }
    create_flow_visualization_video(model_name, FlowNet3DWithFeatureExtraction, 
                                  video_type=video_type, config=config) 