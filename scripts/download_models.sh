#!/bin/bash

mkdir -p models
wget https://zenodo.org/record/10688201/files/retrobridge.ckpt?download=1 -O models/retrobridge.ckpt
wget https://zenodo.org/record/10688201/files/digress.ckpt?download=1 -O models/digress.ckpt
wget https://zenodo.org/record/10688201/files/forwardbridge.ckpt?download=1 -O models/forwardbridge.ckpt