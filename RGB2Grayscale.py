import cv2
# 영상 불러오기

folder_path = "C:/Users/dudgb2380/Downloads/naturalistic_video"
bird_video_path = f"{folder_path}/01_Bird.mp4"
city_video_path = f"{folder_path}/02_City.mp4"
forest_video_path = f"{folder_path}/03_Forest.mp4"

video_list = [bird_video_path, city_video_path]


for video_name in video_list:
    # 동영상 파일 경로
    video_path = video_name

    # 동영상 불러오기
    video = cv2.VideoCapture(video_path)

    # 변환된 동영상 저장을 위한 객체 생성
    output_path = video_path[:video_path.rfind('.')]+'_grayscale_video.avi'
    fps = int(video.get(cv2.CAP_PROP_FPS))
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    new_width = frame_width // 4
    new_height = frame_height // 4
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height), isColor=False)

    while True:
        # 동영상에서 프레임 읽기
        ret, frame = video.read()
        
        # 프레임을 그레이스케일로 변환
        if ret:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized_frame = cv2.resize(gray_frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # 결과 프레임 저장
            out.write(resized_frame)
            
        else:
            break

    # 객체 해제
    video.release()
    out.release()
    cv2.destroyAllWindows()



# 영상 grayscale 변환


# grayscale 영상 저장

