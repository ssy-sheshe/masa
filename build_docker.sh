DOCKER_ADDRESS=registry.aibee.cn/aibee/mmdetection/pytorch1.7.1-py38-cuda11.0-cudnn8-mmdetection_cluster:masa
sudo docker pull $DOCKER_ADDRESS

sudo nvidia-docker run -it \
  --shm-size 64G \
  --ipc=host \
  -p 2766:22 \
  -e COLUMNS=`tput cols` \
  -e LINES=`tput lines` \
  -v /etc/localtime:/etc/localtime:ro \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v /usr/bin/docker:/usr/bin/docker \
  -v /ssd:/ssd \
  -v /mnt:/mnt \
  -v $PWD:/workspace \
  -w /workspace \
  $DOCKER_ADDRESS \
  bash
