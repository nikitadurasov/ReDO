#!/bin/bash
python train.py --dataset flowers --nfX 32 --useSelfAttG --useSelfAttD --outf ./checkpoints/flowers/regularized_mask_mean_concrete_pdf_30_and_base_random_seed \
                --dataroot ./data/flowers --clean --autoRestart 0.1 --batchSize 18 --reg_mask=30 \
                --reg_type mean_concrete_pdf_regularization --reg_max_iteration 5000 \
                | tee ./results/flowers/regularized_mask_mean_concrete_pdf_30_and_base_random_seed.txt
