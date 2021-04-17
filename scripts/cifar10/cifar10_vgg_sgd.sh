#!/bin/bash
epochs=10
device=$@
for lr in 1e-3 1e-2 1e-1
do
	python main.py --dataset cifar10  --batch_size 128  --device $device --optimizer sgd --network vgg16_bn  --epoch $epochs --milestone 10,20,30,40,50 --learning_rate  $lr --learning_rate_decay 0.5  --weight_decay 0.003 --momentum 0.9
done
