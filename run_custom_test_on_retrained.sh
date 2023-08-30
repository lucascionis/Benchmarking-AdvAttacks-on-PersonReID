#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <model_name> <num_iterations>"
    exit 1
fi

model_name="$1"
q="$2"

for ((i=1; i<=q; i++))
do
    echo "Running iteration $i"
    python custom_test.py \
        --targetmodel="$model_name" \
        --dataset='market1501' \
        --G_resume_dir="/content/drive/MyDrive/adv_reid/adv_personreid_results/$model_name/market1501/best_G.pth.tar" \
        --mode='test' \
        --loss='xent_htri' \
        --ak_type=-1 \
        --temperature=-1 \
        --use_SSIM=2 \
        --epoch=40 \
        --max_batches=10 \
        --gallery_batch=32 \
        --top_k=50 \
        --pre_dir="./retrained/retrained_${model_name}_q${i}.pth.tar"
done
