#!/bin/bash

epochs=60
device=$@

# ---------
# SKFAC-S4

## CIFAR-10

### mnv2
for i in 1
do
	python main.py --subsample true --num_ss_patches 4 --dataset cifar10  --batch_size 128 --device cuda --optimizer skfac --network mobilenetv2  --epoch 60 --milestone 10,20,30,40,50 --learning_rate  1e-3 --learning_rate_decay 0.5  --damping 0.3  --weight_decay 0.03 --momentum 0.9
done

### densenet
for i in 1
do
	python main.py --subsample true --num_ss_patches 4 --dataset cifar10  --batch_size 128 --device cuda --optimizer skfac --network densenet  --depth 19 --growthRate 100  --epoch 60 --milestone 10,20,30,40,50 --learning_rate  1e-2 --learning_rate_decay 0.5  --damping 0.3  --weight_decay 0.003 --momentum 0.9
done

### wrn
for i in 1
do
	python main.py --subsample true --num_ss_patches 4 --dataset cifar10  --batch_size 128 --device cuda --optimizer skfac --network wrn  --depth 28 --widen_factor 4  --epoch 60 --milestone 10,20,30,40,50 --learning_rate  3e-2 --learning_rate_decay 0.5  --damping 0.3  --weight_decay 0.001 --momentum 0.9
done

# for i in 1
# do
# 	python main.py --subsample true --num_ss_patches 4 --dataset cifar10  --batch_size 128 --device cuda --optimizer skfac --network wrn  --depth 28 --widen_factor 4  --epoch 60 --milestone 10,20,30,40,50 --learning_rate  3e-3 --learning_rate_decay 0.5  --damping 0.3  --weight_decay 0.03 --momentum 0.9
# done

## CIFAR-100

### mnv2
for i in 1
do
	python main.py --subsample true --num_ss_patches 4 --dataset cifar100  --batch_size 128 --device cuda --optimizer skfac --network mobilenetv2  --epoch 60 --milestone 30,45 --learning_rate  0.003 --learning_rate_decay 0.1  --damping 0.3  --weight_decay 0.003 --momentum 0.9
done

### densenet
for i in 1
do
	python main.py --subsample true --num_ss_patches 4 --dataset cifar100  --batch_size 128 --device cuda --optimizer skfac --network densenet  --depth 19 --growthRate 100  --epoch 60 --milestone 30,45 --learning_rate  0.03 --learning_rate_decay 0.1  --damping 0.3  --weight_decay 0.001 --momentum 0.9
done

### wrn
for i in 1
do
	python main.py --subsample true --num_ss_patches 4 --dataset cifar100  --batch_size 128 --device cuda --optimizer skfac --network wrn  --depth 28 --widen_factor 4  --epoch 60 --milestone 30,45 --learning_rate  0.03 --learning_rate_decay 0.1  --damping 0.3  --weight_decay 0.001 --momentum 0.9
done

# for i in 1
# do
# 	python main.py --subsample true --num_ss_patches 4 --dataset cifar100  --batch_size 128 --device cuda --optimizer skfac --network wrn  --depth 28 --widen_factor 4  --epoch 60 --milestone 30,45 --learning_rate  0.01 --learning_rate_decay 0.1  --damping 0.3  --weight_decay 0.003 --momentum 0.9
# done