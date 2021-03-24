# python main.py --dataset cifar10 --device cpu --optimizer ngd --network vgg16_bn  --epoch 100 --bn true --milestone 40,80 --learning_rate 0.01 --damping 0.03 --weight_decay 0.003
python main.py --dataset cifar10 --device cpu --momentum 0.9 --warmup 500 --optimizer ngd --network vgg16  --epoch 100  --milestone 40,80 --learning_rate 0.01 --damping 0.03 --weight_decay 0.003
