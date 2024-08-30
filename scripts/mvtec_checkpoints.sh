#!/usr/bin/env bash
cd ..
mkdir experiments
cd experiments
mkdir transfusion_mvtec
cd transfusion_mvtec
mkdir models
cd models
pip install gdown
gdown https://docs.google.com/uc?id=1KL7_AwO2zKOHfBY5wcMy_Qj49guSQBc4
unzip transfusion_mvtec.zip