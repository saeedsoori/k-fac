device=$@
python main_autoencoder.py --network autoencoder --dataset mnist --device $device --optimizer ngd  --epoch 5 --learning_rate 0.3 --learning_rate_decay 0.5 --damping 0.3 --stat_decay 0.9 --weight_decay 0 --TCov 1 --TInv 20 --momentum 0.9 --batch_size 1000

epochs=100
device=$@


for lr in 1 3 1e-1 3e-1 1e-2 3e-2 1e-3 3e-3 
do
	for damping in 1 3 1e-1 3e-1 1e-2 3e-2 1e-3 3e-3
	do
    	python main_autoencoder.py --network autoencoder --dataset mnist --device $device --optimizer ngd  --epoch $epochs --learning_rate $lr --learning_rate_decay 0.5 --damping $damping --stat_decay 0.9 --weight_decay 0 --TCov 1 --TInv 20 --momentum 0.9 --batch_size 100
	done
done
