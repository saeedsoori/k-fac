device=$@
python main_autoencoder.py --network autoencoder --dataset mnist --device $device --optimizer kbfgs  --epoch 400 --learning_rate 0.3 --learning_rate_decay 0.5 --damping 0.3 --stat_decay 0.9 --weight_decay 0 --TCov 1 --TInv 1 --momentum 0.9 --batch_size 1000
python main_autoencoder.py --network autoencoder --dataset mnist --device $device --optimizer kbfgs  --epoch 60 --learning_rate 0.1 --learning_rate_decay 0.5 --damping 0.3 --stat_decay 0.9 --weight_decay 0 --TCov 1 --TInv 1 --momentum 0.9 --batch_size 100
