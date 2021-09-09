#!/bin/bash
epochs=60
opt=$@

python main.py --freq 100 --trial true --reduce_sum true --batchnorm false --dataset cifar10  --batch_size 128 --device cuda --optimizer $opt --network wrn  --depth 28 --widen_factor 4  --epoch $epochs --milestone 10,20,30,40,50 --learning_rate  3e-2 --learning_rate_decay 0.5  --damping 0.1  --weight_decay 0.003 --momentum 0.9
# python main.py --freq 100 --trial true --reduce_sum true --batchnorm false --dataset cifar10 --perturb true  --batch_size 128 --device cuda --optimizer $opt --network vgg16_bn   --epoch $epochs --milestone 10,20,30,40,50 --learning_rate  3e-2 --learning_rate_decay 0.5  --damping 0.1  --weight_decay 0.003 --momentum 0.9
# python main.py --freq 100 --trial true --reduce_sum true --batchnorm false --dataset cifar10 --perturb true  --batch_size 128 --device cuda --optimizer $opt --network vgg16_bn   --epoch $epochs --milestone 10,20,30,40,50 --learning_rate  3e-2 --learning_rate_decay 0.5  --damping 0.1  --weight_decay 0.003 --momentum 0.9
# 
# python main.py --freq 100 --network toy --dataset fashion-mnist --learning_rate 0.03 --reduce_sum true --step_info false --damping 0.3 --momentum 0.9 --device cuda --optimizer $opt --epoch $epochs --batch_size 128 --learning_rate_decay 0.5 --milestone 10,20,30,40,50 --save_inv true
# python main.py --freq 100 --trial true --reduce_sum true --batchnorm false --dataset cifar10  --batch_size 128  --device cuda --optimizer $opt --network mobilenetv2  --epoch $epochs --milestone 10,20,30,40,50 --learning_rate  1e-2 --learning_rate_decay 0.5  --damping 0.2  --weight_decay 0.01 --momentum 0.9
