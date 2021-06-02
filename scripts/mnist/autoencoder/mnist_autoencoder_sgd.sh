device=$@
python main_autoencoder.py --network autoencoder --dataset mnist --device $device --optimizer sgd  --epoch 400 --learning_rate 0.03 --learning_rate_decay 0.5 --weight_decay 0 --momentum 0.9 --batch_size 1000
python main_autoencoder.py --network autoencoder --dataset mnist --device $device --optimizer sgd  --epoch 60 --learning_rate 0.03 --learning_rate_decay 0.5 --weight_decay 0 --momentum 0.9 --batch_size 100
