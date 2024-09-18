FILE_NAME=$1
sudo docker exec torch_container bash -c "pkill -f ${FILE_NAME}"