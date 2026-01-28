#!/bin/bash
sudo xhost +si:localuser:root
docker exec -it --privileged vmf-app-1 /bin/bash
