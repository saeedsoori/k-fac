#!/bin/bash
epochs=10
device=$@

echo 'SGD Started'
for i in 1 2 3 4 5
do
	python main.py --network convnet --dataset fashion-mnist --learning_rate 0.03 --step_info true --momentum 0.9  --device $device --optimizer sgd --epoch $epochs --milestone 10,20,30,40,50 --weight_decay 0.001 --batch_size 128
done

echo 'NGD Started'
for i in 1 2 3 4 5
do
   	python main.py --network convnet --dataset fashion-mnist --learning_rate 0.003 --step_info true --momentum 0.9 --damping 0.1 --device $device --optimizer ngd --epoch $epochs --milestone 10,20,30,40,50 --weight_decay 0.001 --batch_size 128
done

echo 'KFAC Started'
for i in 1 2 3 4 5
do 
   	python main.py --network convnet --dataset fashion-mnist --learning_rate 0.003 --step_info true --momentum 0.9 --damping 0.1 --device $device --optimizer kfac --epoch $epochs --milestone 10,20,30,40,50 --weight_decay 0.001 --batch_size 128
done

echo 'EKFAC Started'
for i in 1 2 3 4 5
do 
	python main.py --network convnet --dataset fashion-mnist --learning_rate 0.001 --step_info true --momentum 0.9 --damping 0.03 --device $device --optimizer ekfac --epoch $epochs --milestone 10,20,30,40,50 --weight_decay 0.003 --batch_size 128
done

echo 'KBFGSL Started'
for i in 1 2 3 4 5
do 
	python main.py --network convnet --dataset fashion-mnist --learning_rate 0.01 --step_info true --momentum 0.9 --damping 0.03 --device $device --optimizer kbfgsl --epoch $epochs --milestone 10,20,30,40,50 --weight_decay 0.001 --batch_size 128
done