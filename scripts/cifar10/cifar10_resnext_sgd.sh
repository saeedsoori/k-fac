#!/bin/bash
epochs=60
device=$@
# range lr : 1e-3 1e-2 1e-1 opt:1e-2
for lr in 1e-1 1e-2
do
	python main.py --dataset cifar10  --batch_size 128  --device $device --optimizer sgd --network resnext  --depth 19 --cardinality 1 --widen_factor 4 --base_width 64 --epoch $epochs --milestone 10,20,30,40,50 --learning_rate  $lr --learning_rate_decay 0.5  --weight_decay 0.003 --momentum 0.9
done
