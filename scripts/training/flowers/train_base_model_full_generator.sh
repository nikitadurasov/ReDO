#!/bin/bash
python train.py --dataset flowers --nfX 32 --useSelfAttG --useSelfAttD --outf ./checkpoints/flowers/base_full_generator \
                --dataroot ./data/flowers --clean --manualSeed 42 --autoRestart 0.1 --batchSize 18 --full_generator | tee ./results/flowers/base_full_generator.txt
