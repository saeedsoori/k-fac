# python main.py --network fc --dataset mnist --learning_rate 0.1 --step_info true --momentum 0.9 --damping 0.3 --device cpu --optimizer kbfgsl  --epoch 5  --weight_decay 0.003 --batch_size 128
#!/bin/bash
epochs=10
device=$@
for lr in 1e-3 3e-3 1e-2 3e-2 1e-1 3e-1 1 3 10
do
	for damping in 3e-3 1e-2 3e-2 0.1 0.3 1 3
	do
   		python main.py --network fc --dataset mnist --learning_rate $lr --step_info true --momentum 0.9 --damping $damping --device $device --optimizer kbfgsl  --epoch $epochs   --weight_decay 0.003 --batch_size 128 --num_s_y_pairs 100
	done
done
