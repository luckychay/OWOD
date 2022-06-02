docker run --gpus all -it \
	--shm-size=8gb --env="DISPLAY" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
        --mount type=bind,source=/home/chen/OWOD/output,target=/home/appuser/OWOD/output \
        --name=owod luckychay/owod:v1
     
