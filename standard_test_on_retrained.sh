#!/bin/bash

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <model_name> <num_iterations> <epochs>"
    exit 1
fi

model_name="$1"
q="$2"
epochs="$3"

for ((i=1; i<=q; i++))
do
    echo "Running iteration $i"
    python train.py \
        --targetmodel="$model_name" \
        --dataset='market1501' \
        --G_resume_dir="/content/drive/MyDrive/adv_reid/adv_personreid_results/$model_name/market1501/best_G.pth.tar" \
        --mode='test' \
        --loss='xent_htri' \
        --ak_type=-1 \
        --temperature=-1 \
        --use_SSIM=2 \
        --epoch="$epochs" \
        --pre_dir="./retrained/retrained_${model_name}_q${i}.pth.tar"
done
