device=$@
python main_autoencoder.py --network autoencoder --dataset mnist --device $device --optimizer sgd  --epoch 5 --learning_rate 0.01 --weight_decay 0 --momentum 0.9 --batch_size 1000
