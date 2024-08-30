#!/usr/bin/env bash
cd ..
mkdir experiments
cd experiments
mkdir transfusion_mvtec3d
cd transfusion_mvtec3d
mkdir models
cd models
pip install gdown
gdown https://docs.google.com/uc?id=1RZBZqo-lbywk2lPxnr_H--ERFxbCeXqP
unzip transfusion_mvtec3d.zip