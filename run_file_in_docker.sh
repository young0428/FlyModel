FILE_NAME=$1
sudo docker exec -it torch_container bash -c "cd /myhome/FlyModel && python3 ${FILE_NAME}"
#sudo docker exec -d torch_container bash -c "cd /myhome/FlyModel && python3 ${FILE_NAME} | tee output.log"