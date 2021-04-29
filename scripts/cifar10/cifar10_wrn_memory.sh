#!/bin/bash

epochs=5
device=$@
# SGD
echo 'SGD Started'
python main.py --step_info true --debug_mem true --dataset cifar10  --batch_size 128  --device $device --optimizer sgd --network wrn  --depth 28 --widen_factor 4  --epoch $epochs --milestone 10,20,30,40,50 --learning_rate  1e-2 --learning_rate_decay 0.5  --weight_decay 0.003 --momentum 0.9
echo 'SGD finished'

# NGD
echo 'NGD Started'
python main.py --step_info true --debug_mem true --freq 20 --trial true  --batchnorm false --dataset cifar10  --batch_size 128  --device $device --optimizer ngd --network wrn  --depth 28 --widen_factor 4  --epoch $epochs --milestone 10,20,30,40,50 --learning_rate  3e-2 --learning_rate_decay 0.5  --damping 0.2  --weight_decay 0.003 --momentum 0.9
echo 'NGD Finished'

# KFAC
echo 'KFAC Started'
python main.py --step_info true --debug_mem true --TInv 20 --dataset cifar10  --batch_size 128 --device $device --optimizer kfac --network wrn  --depth 28 --widen_factor 4  --epoch $epochs --milestone 10,20,30,40,50 --learning_rate  1e-2 --learning_rate_decay 0.5  --damping 0.1  --weight_decay 0.003 --momentum 0.9
echo 'KFAC Finsihed'

# KFAC
echo 'EKFAC Started'
python main.py --step_info true --debug_mem true --TInv 20 --dataset cifar10  --batch_size 128 --device $device --optimizer ekfac   --network wrn  --depth 28 --widen_factor 4  --epoch $epochs --milestone 10,20,30,40,50 --learning_rate  1e-2 --learning_rate_decay 0.5  --damping 0.1  --weight_decay 0.003 --momentum 0.9
echo 'EKFAC Finsihed'

# KFAC
echo 'KBFGS-L Started'
python main.py --step_info true --debug_mem true --TInv 20 --dataset cifar10  --batch_size 128 --device $device --optimizer kbfgsl --network wrn  --depth 28 --widen_factor 4  --epoch $epochs --milestone 10,20,30,40,50 --learning_rate  1e-2 --learning_rate_decay 0.5  --damping 0.1  --weight_decay 0.003 --momentum 0.9
echo 'KBFGS-L Finsihed'
