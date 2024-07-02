import os
import time
import cv2
import numpy as np
import threading
import keyboard

class OpticalFlowDisplay:
    def __init__(self, video_path, output_path_upward, output_path_downward, output_path_rightward, output_path_leftward, output_path_total, output_path_combined):
        self.cap = cv2.VideoCapture(video_path)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.display_flag = True
        self.stop_flag = False
        self.video_path = video_path

        # 화면 크기 설정
        self.screen_width = 1920
        self.screen_height = 1080
        self.display_width = self.screen_width // 2
        self.display_height = self.screen_height // 2

        # 다운스케일 크기
        self.downscale_width = self.width // 4
        self.downscale_height = self.height // 4

        # 비디오 라이터 객체 생성
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.out_upward = cv2.VideoWriter(output_path_upward, fourcc, self.fps, (self.downscale_width, self.downscale_height), isColor=True)
        self.out_downward = cv2.VideoWriter(output_path_downward, fourcc, self.fps, (self.downscale_width, self.downscale_height), isColor=True)
        self.out_rightward = cv2.VideoWriter(output_path_rightward, fourcc, self.fps, (self.downscale_width, self.downscale_height), isColor=True)
        self.out_leftward = cv2.VideoWriter(output_path_leftward, fourcc, self.fps, (self.downscale_width, self.downscale_height), isColor=True)
        self.out_total = cv2.VideoWriter(output_path_total, fourcc, self.fps, (self.downscale_width, self.downscale_height), isColor=True)
        self.out_combined = cv2.VideoWriter(output_path_combined, fourcc, self.fps, (self.display_width, self.display_height), isColor=True)

    def display_video(self):
        frame_count = 0
        ret, first_frame = self.cap.read()
        if not ret:
            print("비디오를 읽을 수 없습니다.")
            self.cap.release()
            return

        prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

        while self.cap.isOpened() and not self.stop_flag:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame_count += 1
            progress = (frame_count / self.total_frames) * 100
            print(f"\r{os.path.basename(self.video_path)} Progress: {progress:.2f}%", end="")

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 
                                                pyr_scale=0.3, levels=5, winsize=21, 
                                                iterations=5, poly_n=7, poly_sigma=1.5, flags=0)
            flow_x = flow[..., 0]
            flow_y = flow[..., 1]
            upward_mag = np.where(flow_y > 0, flow_y, 0)
            downward_mag = np.where(flow_y < 0, -flow_y, 0)
            rightward_mag = np.where(flow_x > 0, flow_x, 0)
            leftward_mag = np.where(flow_x < 0, -flow_x, 0)
            total_mag = np.sqrt(flow_x**2 + flow_y**2)

            grayscale_upward = cv2.normalize(upward_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            grayscale_downward = cv2.normalize(downward_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            grayscale_rightward = cv2.normalize(rightward_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            grayscale_leftward = cv2.normalize(leftward_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            grayscale_total = cv2.normalize(total_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
              

            grayscale_upward_color = cv2.cvtColor(grayscale_upward, cv2.COLOR_GRAY2BGR)
            grayscale_downward_color = cv2.cvtColor(grayscale_downward, cv2.COLOR_GRAY2BGR)
            grayscale_rightward_color = cv2.cvtColor(grayscale_rightward, cv2.COLOR_GRAY2BGR)
            grayscale_leftward_color = cv2.cvtColor(grayscale_leftward, cv2.COLOR_GRAY2BGR)
            grayscale_total_color = cv2.cvtColor(grayscale_total, cv2.COLOR_GRAY2BGR)
            
            grayscale_upward_color = cv2.resize(grayscale_upward_color, (self.downscale_width, self.downscale_height))
            grayscale_downward_color = cv2.resize(grayscale_downward_color, (self.downscale_width, self.downscale_height))
            grayscale_rightward_color = cv2.resize(grayscale_rightward_color, (self.downscale_width, self.downscale_height))
            grayscale_leftward_color = cv2.resize(grayscale_leftward_color, (self.downscale_width, self.downscale_height))
            grayscale_total_color = cv2.resize(grayscale_total_color, (self.downscale_width, self.downscale_height))
            frame_downscaled = cv2.resize(frame, (self.downscale_width, self.downscale_height))


            top_row = np.hstack((frame_downscaled, grayscale_upward_color, grayscale_downward_color))
            bottom_row = np.hstack((grayscale_rightward_color, grayscale_leftward_color, grayscale_total_color))
            combined_image = np.vstack((top_row, bottom_row))

            resized_combined_image = cv2.resize(combined_image, (self.display_width, self.display_height))
            
            self.out_upward.write(grayscale_upward_color)
            self.out_downward.write(grayscale_downward_color)
            self.out_rightward.write(grayscale_rightward_color)
            self.out_leftward.write(grayscale_leftward_color)
            self.out_total.write(grayscale_total_color)
            
            self.out_combined.write(resized_combined_image)

            if self.display_flag:
                cv2.imshow('Optical Flow', resized_combined_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.display_flag = False
                    cv2.destroyWindow('Optical Flow')
            else:
                cv2.destroyAllWindows()

            prev_gray = gray

        self.cap.release()
        self.out_upward.release()
        self.out_downward.release()
        self.out_rightward.release()
        self.out_leftward.release()
        self.out_total.release()
        self.out_combined.release()
        self.stop_flag = True
        cv2.destroyAllWindows()
        print()  # 진행 표시를 완료한 후 줄 바꿈

    def toggle_display(self):
        while not self.stop_flag:
            
            if keyboard.is_pressed('q'):
                self.display_flag = False
                
            elif keyboard.is_pressed('r'):
                self.display_flag = True
                
            time.sleep(0.5)

def run_optical_flow_display(video_path, output_path_upward, output_path_downward, output_path_rightward, output_path_leftward, output_path_total, output_path_combined):
    optical_flow_display = OpticalFlowDisplay(video_path, output_path_upward, output_path_downward, output_path_rightward, output_path_leftward, output_path_total, output_path_combined)
    display_thread = threading.Thread(target=optical_flow_display.display_video)
    toggle_thread = threading.Thread(target=optical_flow_display.toggle_display)

    display_thread.start()
    toggle_thread.start()

    display_thread.join()
    toggle_thread.join()    
folder_path = "C:/Users/dudgb2380/Downloads/naturalistic_video"
bird_video_path = f"{folder_path}/01_Bird.mp4"
city_video_path = f"{folder_path}/02_City.mp4"
forest_video_path = f"{folder_path}/03_Forest.mp4"

video_list = [city_video_path, forest_video_path]

for video_path in video_list:
    output_path_upward = f"{folder_path}/direction/{os.path.basename(video_path)}_upward.avi"
    output_path_downward = f"{folder_path}/direction/{os.path.basename(video_path)}_downard.avi"
    output_path_rightward = f"{folder_path}/direction/{os.path.basename(video_path)}_rightward.avi"
    output_path_leftward = f"{folder_path}/direction/{os.path.basename(video_path)}_leftward.avi"
    output_path_total = f"{folder_path}/direction/{os.path.basename(video_path)}_total.avi"
    output_path_combined = f"{folder_path}/direction/{os.path.basename(video_path)}_combined.avi"

    run_optical_flow_display(video_path, output_path_upward, output_path_downward, output_path_rightward, output_path_leftward, output_path_total,output_path_combined)
