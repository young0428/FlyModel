import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import torch
from FlowNet import *
from trainer_func import *
from LoadDataset import *
import pickle
import os

def get_frame_per_window(model_path):
    """모델 경로에서 frame_per_window 값을 추출"""
    # 경로를 '/'로 분리
    parts = model_path.split('/')
    
    for part in parts:
        # '_frames' 또는 'frames'가 포함된 부분 찾기
        if 'frames' in part:
            # 숫자 추출
            frame_num = int(''.join(filter(str.isdigit, part)))
            return frame_num
    
    return 8  # 기본값 반환

def load_model_and_data(model_name, piece_size):
    """모델과 관련 데이터를 로드하는 함수"""
    # 먼저 기본 경로에서 frame_per_window 찾기
    base_paths = [
        f"./model/{model_name}",  # 기본 경로
        f"./model/{model_name}_8frames",  # 8frames가 포함된 경로
        f"./model/{model_name}_16frames"  # 16frames가 포함된 경로
    ]
    
    model_path = None
    for base_path in base_paths:
        temp_path = f"{base_path}/piece_size_{piece_size}_fix_True/fold_1"
        if os.path.exists(temp_path):
            model_path = temp_path
            break
    
    if model_path is None:
        raise FileNotFoundError(f"Could not find model path for piece_size {piece_size}")
    
    frame_per_window = get_frame_per_window(model_path)
    
    # 모델 로드
    flownet_model = flownet3d([[64, 2], [128, 2], [256, 2]])
    model = FlowNet3DWithFeatureExtraction(flownet_model, feature_dim=128, 
                                         input_size=(frame_per_window, 64, 128, 1))
    trainer = Trainer(model, loss_function_mse, 1e-4)
    trainer.load(f"{model_path}/best_model.ckpt")
    
    # 튜플 데이터 로드
    with open(f"{model_path}/training_tuples.pkl", 'rb') as f:
        train_tuples = pickle.load(f)
    with open(f"{model_path}/validation_tuples.pkl", 'rb') as f:
        val_tuples = pickle.load(f)
        
    return trainer, train_tuples, val_tuples, frame_per_window

def update(frame, ax_video, ax_graphs, trainers, train_tuples_list, val_tuples_list, 
          video_data, wba_data, frame_per_windows, prev_predictions, prediction_lines, piece_sizes):
    # 비디오 프레임 업데이트
    video_frame = video_data[2, frame, :, :, 0]
    ax_video.clear()
    ax_video.imshow(video_frame, cmap='gray')
    ax_video.axis('off')
    
    # 각 그래프 업데이트
    for i, (ax, trainer, train_tuples, val_tuples, fpw, piece_size) in enumerate(
        zip(ax_graphs, trainers, train_tuples_list, val_tuples_list, frame_per_windows, piece_sizes)):
        
        # 그래프 초기화 (하얀 배경 유지)
        ax.clear()
        ax.plot(wba_data[2], color='gray', alpha=0.3)
        ax.set_title(f'Piece Size: {piece_size} (Window: {fpw} frames)')
        ax.set_ylim(-10, 30)
        
        # x축 틱 설정
        if i == len(piece_sizes) - 1:  # 마지막 그래프
            ax.tick_params(axis='x', labelbottom=True)  # x축 레이블 표시
        else:
            ax.tick_params(axis='x', labelbottom=False)  # x축 레이블 숨기기
        
        # 이전 예측선들 다시 그리기
        for prev_frame, (x, y, color) in prediction_lines[i].items():
            if prev_frame <= frame:  # 현재 프레임까지만 그리기
                ax.plot(x, y, color=color, linewidth=1)
        
        # 현재 프레임 위치 표시
        ax.axvline(x=frame, color='black', linestyle='-', alpha=0.5)
        
        # train/val 튜플에서 현재 프레임이 정확히 일치하는지 확인
        is_train = any(tup[1] == frame for tup in train_tuples)
        is_val = any(tup[1] == frame for tup in val_tuples)
        
        # train 또는 val 프레임과 정확히 일치할 때만 예측 수행
        if (is_train or is_val) and frame >= fpw:
            input_data = torch.tensor(
                video_data[2:3, frame-fpw:frame, :, :, 0:1], 
                dtype=torch.float32
            ).to(trainer.device)
            
            # 이전 예측값이 있으면 사용, 없으면 실제 WBA 값 사용
            if frame-fpw in prev_predictions[i]:
                prev_wba = prev_predictions[i][frame-fpw]
            else:
                prev_wba = wba_data[2, frame-fpw]
                
            wba_input = torch.tensor(
                [[prev_wba]], 
                dtype=torch.float32
            ).to(trainer.device)
            
            with torch.no_grad():
                pred = trainer.model(input_data, wba_input)
                # 예측값 저장
                prev_predictions[i][frame] = pred.item()
                
            color = 'blue' if is_train else 'red'
            x_coords = [frame-fpw, frame]
            y_coords = [prev_wba, pred.item()]
            ax.plot(x_coords, y_coords, color=color, linewidth=1)
            
            # 예측선 저장
            prediction_lines[i][frame] = (x_coords, y_coords, color)

def create_visualization_video(model_name, piece_sizes=[1, 5, 10, 20, 40]):
    """전체 시각화 동영상을 생성하는 메인 함수"""
    # 데이터 로드
    video_data, wba_data, total_frame = direction_pred_training_data_preparing_seq(
        "./naturalistic", "experimental_data.mat", 5.625)
    
    # 그래프 설정
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(len(piece_sizes) + 1, 1, height_ratios=[2] + [1]*len(piece_sizes))
    
    # 비디오 표시 영역
    ax_video = fig.add_subplot(gs[0])
    ax_video.axis('off')
    
    # 그래프 영역
    ax_graphs = []
    trainers = []
    train_tuples_list = []
    val_tuples_list = []
    frame_per_windows = []
    
    for i, piece_size in enumerate(piece_sizes):
        ax_graphs.append(fig.add_subplot(gs[i+1]))
        trainer, train_tuples, val_tuples, fpw = load_model_and_data(model_name, piece_size)
        trainers.append(trainer)
        train_tuples_list.append(train_tuples)
        val_tuples_list.append(val_tuples)
        frame_per_windows.append(fpw)
        
        # 실제 WBA 데이터 플롯
        ax_graphs[-1].plot(wba_data[2], color='gray', alpha=0.3)
        ax_graphs[-1].set_title(f'Piece Size: {piece_size} (Window: {fpw} frames)')
    
    frame_size = (1500, 1000)  # 현재 프레임 크기
    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # AVI 포맷 사용
    output_filename = 'wba_value_whole_features_visualization.avi'  # 확장자를 .avi로 변경

    # 비디오 writer 객체 생성
    out = cv2.VideoWriter(output_filename, fourcc, fps, frame_size, isColor=True)

    # 이전 예측값과 예측선을 저장할 딕셔너리 리스트 초기화
    prev_predictions = [{} for _ in piece_sizes]
    prediction_lines = [{} for _ in piece_sizes]
    
    try:
        plt.show(block=False)  # 창을 먼저 띄우기
        
        # 애니메이션의 각 프레임을 저장
        for frame in range(0, total_frame-3):
            # 프레임 업데이트
            update(frame, ax_video, ax_graphs, trainers, train_tuples_list, val_tuples_list, 
                   video_data, wba_data, frame_per_windows, 
                   prev_predictions, prediction_lines, piece_sizes)
            
            # 실시간으로 plot 업데이트
            plt.pause(0.001)  # 화면 업데이트를 위한 짧은 일시 정지
            
            # matplotlib figure를 이미지로 변환
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            # 프레임 저장
            out.write(img)
            
            if frame % 100 == 0:  # 진행상황 출력
                print(f"프레임 {frame}/{total_frame-3} 처리 중...")
        
        print(f"성공적으로 저장되었습니다: {output_filename}")

    except Exception as e:
        print(f"비디오 저장 실패: {str(e)}")
        raise

    finally:
        # 리소스 해제
        out.release()
        cv2.destroyAllWindows()
        plt.close()

    plt.close()

# 사용 예시
if __name__ == "__main__":
    model_name = "wba_value_whole_features"
    create_visualization_video(model_name)