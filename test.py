import os
import shutil
import re

# 모델 폴더 경로
model_dir = './model'

# 모델 폴더 내의 모든 폴더를 가져옵니다.
folders = [f for f in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, f))]

# 숫자를 제외한 문자열을 기준으로 정리합니다.
for folder in folders:
    # 숫자를 제외한 문자열을 추출합니다.
    common_name = re.sub(r'\d+', '', folder).rstrip('_')

    # 공통된 이름의 폴더 경로를 만듭니다.
    common_folder_path = os.path.join(model_dir, common_name)

    # 폴더가 자기 자신 안으로 이동되지 않도록 확인합니다.
    if common_folder_path != os.path.join(model_dir, folder):
        os.makedirs(common_folder_path, exist_ok=True)
        shutil.move(os.path.join(model_dir, folder), os.path.join(common_folder_path, folder))