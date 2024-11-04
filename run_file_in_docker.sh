#!/bin/bash
#SBATCH --job-name=dl_training
#SBATCH --output=output.log
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --partition=p1

FILE_NAME=$1
sudo docker exec -it torch_container bash -c "cd /myhome/FlyModel && python3 ${FILE_NAME} --log_to_file False"
#sudo docker exec -d torch_container bash -c "cd /myhome/FlyModel && python3 ${FILE_NAME} | tee output.log"
#sudo docker exec torch_container bash -c "cd /myhome/FlyModel && ./task_queue.sh 'python3 $FILE_NAME'"