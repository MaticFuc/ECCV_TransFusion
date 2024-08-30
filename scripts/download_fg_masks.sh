#!/usr/bin/env bash
cd ..
mkdir fg_masks
cd fg_masks
pip install gdown
gdown https://docs.google.com/uc?id=1oWAWDnZBcRcdVMyPk4zW0c6pNDmBqbOv
gdown https://docs.google.com/uc?id=1NOJSeAYwfgrQbaQq08B2zL9i9_yXAic2
gdown https://docs.google.com/uc?id=1vMjKhxSH1UdFLIAXu2pr_9UqoknmVUJA
unzip mvtec_masks.zip
unzip mvtec3d_masks.zip
unzip visa_masks.zip
rm mvtec_masks.zip
rm mvtec3d_masks.zip
rm visa_masks.zip