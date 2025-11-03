
#docker create --gpus all --shm-size="10g" --cap-add=SYS_ADMIN \
docker create --gpus all --net=host --shm-size="10g" --cap-add=SYS_ADMIN \
  -v .:/workspace/verl \
  -v $HOME/data:/data \
  -v /local/data/lorena/:/local/data/lorena/ \
  -v /proj/interaction/interaction-filer/lorena:/proj/interaction/interaction-filer/lorena/ \
  --name verl \
  verlai/verl:app-verl0.5-transformers4.55.4-vllm0.10.0-mcore0.13.0-te2.2 sleep infinity

#bash my_scripts/create_docker.sh