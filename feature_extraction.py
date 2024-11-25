import torch
import numpy as np
from FlowNet import flownet3d, FlowNet3DWithFeatureExtraction
from LoadDataset import direction_pred_training_data_preparing_seq
import os
import cv2

class FeatureExtractor(torch.nn.Module):
    def __init__(self, model):
        super(FeatureExtractor, self).__init__()
        self.encoder = model.flownet3d.encoder
        self.decoder = model.flownet3d.decoder
        
    def forward(self, x):
        x = x.permute(0, 4, 1, 2, 3)  # (batch, channel, time, height, width)
        encoder_outputs = self.encoder(x)
        decoder_output = self.decoder(encoder_outputs)
        decoder_output = decoder_output.permute(0, 2, 3, 4, 1)
        
        # for i in range(len(self.decoder.upconvs)):
        #     decoder_output = self.decoder.upconvs[i](decoder_output)
        #     decoder_output = torch.cat([decoder_output, encoder_outputs[-(i+2)]], dim=1)
        #     decoder_output = self.decoder.convs[i](decoder_output)
            
        return decoder_output

def extract_and_save_feature_video(model_config, video_type=2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 모델 경로 구성
    base_path = f"./model/{model_config['name']}"
    config_string = f"piece_size_{model_config['piece_size']}_fix_{model_config['fix']}_pretrained_{model_config['pretrained']}"
    model_path = os.path.join(base_path, config_string, f"fold_{model_config['fold']}")
    
    # 사전학습 모델과 학습된 모델 준비
    flownet_model_pretrained = flownet3d([[64, 2], [128, 2], [256, 2]])
    flownet_model_trained = flownet3d([[64, 2], [128, 2], [256, 2]])
    
    # 사전학습 모델 로드
    pretrained_model = FlowNet3DWithFeatureExtraction(flownet_model_pretrained, feature_dim=128, 
                                                    input_size=(8, 64, 128, 1))
    pretrained_checkpoint = torch.load("./pretrained_model/64_to_256_3layers.ckpt", map_location=device)
    pretrained_model.load_state_dict(pretrained_checkpoint['model_state_dict'], strict=False)
    pretrained_model.to(device)
    pretrained_model.eval()
    
    # 학습된 모델 로드
    trained_model = FlowNet3DWithFeatureExtraction(flownet_model_trained, feature_dim=128, 
                                                 input_size=(8, 64, 128, 1))
    trained_checkpoint = torch.load(f"{model_path}/best_model.ckpt", map_location=device)
    trained_model.load_state_dict(trained_checkpoint['model_state_dict'])
    trained_model.to(device)
    trained_model.eval()
    
    # Feature Extractor 생성
    pretrained_extractor = FeatureExtractor(pretrained_model)
    trained_extractor = FeatureExtractor(trained_model)
    pretrained_extractor.to(device)
    trained_extractor.to(device)
    
    # 비디오 데이터 로드
    video_data, _, _ = direction_pred_training_data_preparing_seq(
        "./naturalistic", 
        "experimental_data.mat", 
        5.625
    )
    
    # window size 설정
    window_size = int(model_config['name'].split('frames')[0][-1])
    
    # 출력 디렉토리 생성
    video_names = ['bird', 'city', 'forest']
    output_dir = f"./extracted_feature/{model_config['name']}/{config_string}"
    os.makedirs(output_dir, exist_ok=True)
    
    # 비디오 writer 설정
    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_filename = f"{output_dir}/{video_names[video_type]}_feature_comparison.mp4"
    
    reconstructed_frames = []
    
    with torch.no_grad():
        video = video_data[video_type:video_type+1]
        
        # window_size 단위로 처리 (no overlap)
        for start_idx in range(0, video.shape[1] - window_size + 1, window_size):
            input_data = torch.tensor(
                video[:, start_idx:start_idx+window_size,:,:,0:1],
                dtype=torch.float32
            ).to(device)
            
            # 사전학습 모델과 학습된 모델의 feature extraction
            pretrained_output = pretrained_extractor(input_data)
            trained_output = trained_extractor(input_data)
            
            # 각 time step의 feature를 프레임으로 변환
            for t in range(pretrained_output.shape[1]):  # time dimension
                # 사전학습 모델의 feature
                pretrained_frame = pretrained_output[0, t, :, :, 0].cpu().numpy() + pretrained_output[0, t, :, :, 1].cpu().numpy()
                pretrained_frame = pretrained_frame * (254/pretrained_frame.max())
                pretrained_frame = pretrained_frame.astype(np.uint8)
                #pretrained_frame = (pretrained_frame * 255).astype(np.uint8)
                
                # 학습된 모델의 feature
                trained_frame = trained_output[0, t, :, :, 0].cpu().numpy() + trained_output[0, t, :, :, 1].cpu().numpy()
                trained_frame = trained_frame * (254/trained_frame.max())
                trained_frame = trained_frame.astype(np.uint8)
                #trained_frame = (trained_frame - trained_frame.min()) / (trained_frame.max() - trained_frame.min())
                #trained_frame = (trained_frame * 255).astype(np.uint8)
                
                # 두 프레임을 가로로 연결
                combined_frame = np.hstack([pretrained_frame, trained_frame])
                reconstructed_frames.append(combined_frame)
            
            if start_idx % 100 == 0:
                print(f"Processing frame {start_idx}/{video.shape[1]}")
    
    # 비디오 저장
    print(np.shape(reconstructed_frames))
    height, width = reconstructed_frames[0].shape
    out = cv2.VideoWriter(video_filename, fourcc, fps, (width, height), False)
    
    for frame in reconstructed_frames:
        # 프레임에 텍스트 추가
        frame_with_text = frame.copy()
        # h, w = frame_with_text.shape
        # cv2.putText(frame_with_text, 'Pretrained', (10, 30), 
        #             cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
        # cv2.putText(frame_with_text, 'Trained', (w//2 + 10, 30), 
        #             cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
        out.write(frame_with_text)
    
    out.release()
    print(f"Feature comparison video saved to {video_filename}")

if __name__ == "__main__":
    model_config = {
        'name': 'city_wba_value_compare_pretrained_and_non_8frames',
        'piece_size': 1,
        'fix': False,
        'pretrained': True,
        'fold': 1
    }
    
    extract_and_save_feature_video(model_config, video_type=2) 