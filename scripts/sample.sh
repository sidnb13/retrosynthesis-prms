#!/bin/bash

# python RetroBridge/sample.py \
#        --config RetroBridge/configs/retrobridge.yaml \
#        --checkpoint models/retrobridge.ckpt \
#        --samples samples \
#        --model RetroBridge \
#        --mode test \
#        --n_samples 10 \
#        --n_steps 250 \
#        --sampling_seed 1

python RetroBridge/sample.py \
       --config RetroBridge/configs/digress.yaml \
       --checkpoint models/digress.ckpt \
       --samples samples \
       --model DiGress \
       --mode test \
       --n_samples 10 \
       --n_steps 500 \
       --sampling_seed 1