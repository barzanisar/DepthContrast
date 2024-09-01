#!/bin/bash

docker build -t ssl:cluster_nuscenes -f Dockerfile_cluster .

# sudo SINGULARITY_NOHTTPS=1 singularity build ssl_latest.sif docker-daemon://ssl:latest

# Push docker image to docker hub
#create a repo in docker hub called barzanisar/ssl
#docker tag ssl:latest barzanisar/ssl:latest
#docker push barzanisar/ssl:latest


#in turing (not enough memory error on lovelace)
#docker pull barzanisar/ssl:latest
#docker images