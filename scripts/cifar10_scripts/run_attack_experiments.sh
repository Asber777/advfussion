#!/bin/bash

# 定义需要修改的参数数组
# scale_arr=(1.25 1.75 2.0 2.25 2.5)
# scale_arr=(3.0 3.25 3.5 3.75 4.0)
# python advfussion/scripts/half_cam_attack.py --use_adver False --adver_scale 0
# scale_arr=(0.75 2.75)
scale_arr=(160 180 200 220 240 260 280 300 350 400 450)
# 循环遍历数组中的元素
for scale in "${scale_arr[@]}"
do
	echo "Running experiment with parameters: $scale"
	# 执行实验启动的脚本，传入命令行参数
	python /root/hhtpro/123/advfussion/scripts/cifar10_scripts/PureGA.py \
	--eval_path /root/hhtpro/123/result_of_cifar10_exp/GA/max_epsilon-$scale \
	--device cuda:0
done
