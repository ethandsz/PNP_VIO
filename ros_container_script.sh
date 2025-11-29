xhost local:root


XAUTH=/tmp/.docker.xauth

docker run -it \
    --name=PNP_VIO \
    --env="DISPLAY=$DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --volume="/home/ethan/Documents/Github/PNP_VIO/:/home/ethan/Documents/Github/PNP_VIO/" \
    --env="XAUTHORITY=$XAUTH" \
    --volume="$XAUTH:$XAUTH" \
    --device=/dev/video4 \
    --net=host \
    --privileged \
    --runtime=nvidia \
    -p 9999:9999 \
    pnp-vio:latest \
    bash

echo "Done."
