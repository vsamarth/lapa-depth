
https://github.com/LatentActionPretraining/LAPA

deactivate 2>/dev/null || true
rm -rf .venv
source .venv/bin/activate

1. Build and run docker 
docker build -t lapa-depth .

docker run --gpus all -it --rm \
  --name lapa_depth_dev \
  -v $PWD:/workspace/lapa \
  -v /media/do/data1/philo/lapa:/datasets \
  -w /workspace/lapa \
  lapa-depth bash

docker exec -it lapa_depth_dev bash

Pack docker image: 
docker save -o lapa-depth.tar lapa-depth:latest

Compress docker image:
gzip lapa-depth.tar

Check file size: 
ls -lh lapa-depth.tar.gz

5. Trên máy mới, load image
gunzip lapa-depth.tar.gz
docker load -i lapa-depth.tar

Check: 
docker images | grep lapa-depth

6. Chạy container trên máy mới
docker run --gpus all -it --rm \
  --name lapa_depth_dev \
  -v $PWD:/workspace/lapa \
  -v /media/do/data1/philo/lapa:/datasets \
  -w /workspace/lapa \
  lapa-depth bash


