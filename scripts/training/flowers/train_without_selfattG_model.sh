#!/bin/bash
python train.py --dataset flowers --nfX 32 --useSelfAttD --outf ./checkpoints/flowers/without_selfattG \
                --dataroot ./data/flowers --clean --manualSeed 42 --autoRestart 0.1 --batchSize 18 | tee ./results/flowers/withou:wqt_selfattG.txt
