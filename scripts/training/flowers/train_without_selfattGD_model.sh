#!/bin/bash
python train.py --dataset flowers --nfX 32 --outf ./checkpoints/flowers/without_selfattGD \
                --dataroot ./data/flowers --clean --manualSeed 42 --autoRestart 0.1 --batchSize 18 | tee ./results/flowers/without_selfattGD.txt
