#!/bin/bash

# 사용법: ./run_docker.sh <ct_name>

# 컨테이너 이름을 첫 번째 인자로 받습니다.
CT_NAME=$1

# 현재 사용자의 홈 디렉토리를 변수에 저장합니다.
HOME_DIR="/home/$USER"

# 도커 실행 명령어
sudo nohup docker run -it --name ${CT_NAME} -v ${HOME_DIR}:/myhome --gpus all pytorch/pytorch:latest

