device=$@
python main_autoencoder.py --network autoencoder --dataset mnist --device $device --optimizer kfac  --epoch 5 --learning_rate 1 --damping 3 --stat_decay 0.9 --weight_decay 0 --TCov 1 --TInv 20 --momentum 0.9 --batch_size 1000 --torch_symeig false
