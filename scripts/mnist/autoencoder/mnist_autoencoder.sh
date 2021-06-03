#!/bin/bash

epochs=100
device=$@
# SGD
echo 'SGD Started'
for i in 1 
do
	python main_autoencoder.py --network autoencoder --dataset mnist --device $device --optimizer sgd  --epoch $epochs --learning_rate 0.003 --learning_rate_decay 1 --weight_decay 0 --momentum 0.9 --batch_size 100
done
echo 'SGD finished'

# NGD
echo 'NGD Started'
for i in 1
do
	python main_autoencoder.py --network autoencoder --dataset mnist --device $device --optimizer ngd  --epoch $epochs --learning_rate 0.1 --learning_rate_decay 1 --damping 3 --stat_decay 0.9 --weight_decay 0 --TCov 20 --freq 100 --momentum 0.9 --batch_size 100
done
echo 'NGD Finished'

# KFAC
echo 'KFAC Started'
for i in 1 
do
	python main_autoencoder.py --network autoencoder --dataset mnist --device $device --optimizer kfac  --epoch $epochs --learning_rate 0.1 --learning_rate_decay 1 --damping 0.3 --stat_decay 0.9 --weight_decay 0 --TCov 20 --TInv 100 --momentum 0.9 --batch_size 100 --torch_symeig false
done
echo 'KFAC Finsihed'

# EKFAC
echo 'EKFAC Started'
for i in 1 
do
	python main_autoencoder.py --network autoencoder --dataset mnist --device $device --optimizer ekfac  --epoch $epochs --learning_rate 0.001 --learning_rate_decay 1 --damping 0.3 --stat_decay 0.9 --weight_decay 0 --TCov 20 --TInv 100 --momentum 0.9 --batch_size 100
done
echo 'EKFAC Finsihed'

# KBFGS
echo 'KBFGS Started'
for i in 1
do
	python main_autoencoder.py --network autoencoder --dataset mnist --device $device --optimizer kbfgs  --epoch $epochs --learning_rate 0.1 --learning_rate_decay 1 --damping 0.3 --stat_decay 0.9 --weight_decay 0 --TCov 1 --TInv 1 --momentum 0.9 --batch_size 100
done
echo 'KBFGS Finsihed'

# # KBFGSL
# echo 'KBFGS-L Started'
# for i in 1 2 3 4 5
# do
# 	python main_autoencoder.py --network autoencoder --dataset mnist --device $device --optimizer kbfgsl  --epoch $epochs --learning_rate 0.1 --learning_rate_decay 1 --damping 0.3 --stat_decay 0.9 --weight_decay 0 --TCov 1 --TInv 1 --momentum 0.9 --batch_size 100 --num_s_y_pairs 100
# done
# echo 'KBFGS-L Finsihed'
