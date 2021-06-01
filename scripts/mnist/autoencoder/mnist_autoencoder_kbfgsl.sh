device=$@
python main_autoencoder.py --network autoencoder --dataset mnist --device $device --optimizer kbfgsl  --epoch 5 --learning_rate 0.3 --damping 0.3 --stat_decay 0.9 --weight_decay 0 --TCov 1 --TInv 1 --momentum 0.9 --batch_size 1000 --num_s_y_pairs 100
