#!/bin/bash
python train.py --dataset flowers --nfX 32 --useSelfAttG --useSelfAttD --outf ./checkpoints/flowers/regularized_mask_10 \
                --dataroot ./data/flowers --clean --manualSeed 42 --autoRestart 0.1 --batchSize 18 --wrecZ=0 --reg_mask=10 \
                --reg_type mean_mse_regularization \
                | tee ./results/flowers/regularized_mask_10.txt
