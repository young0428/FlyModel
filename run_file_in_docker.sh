#!/bin/bash
#SBATCH --job-name=dl_training
#SBATCH --output=output.log
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --partition=p1

FILE_NAME=$1


#!/bin/bash
#SBATCH --job-name=dl_training
#SBATCH --output=output.log
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --partition=p1

FILE_NAME=$1

# GPU 사용 여부를 확인하는 함수
check_gpu_usage() {
    nvidia-smi | grep -q 'No running processes found'
    return $?
}

# GPU가 사용 중일 때 대기
while ! check_gpu_usage; do
    echo "GPU 사용 중입니다. 5분 후 다시 확인합니다."
    sleep 300  # 10분 대기
done


# GPU 사용 여부를 다시 확인
if check_gpu_usage; then
    echo "작업을 시작합니다."
    sudo docker exec -it torch_container bash -c "cd /myhome/FlyModel && python3 ${FILE_NAME}"
    #sudo docker exec -d torch_container bash -c "cd /myhome/FlyModel && python3 ${FILE_NAME} | tee output.log"
else
    echo "다른 사용자가 GPU를 사용 중입니다. 스크립트를 종료합니다."
fi