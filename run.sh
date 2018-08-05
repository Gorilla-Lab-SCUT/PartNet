#!/bin/bash
########################################## Use the following command to test the final results ####################################
python main.py --batch_size 128 --batch_size_partnet 64 --momentum 0.9 --weight_decay 1e-4 --data_path /data/flower102/   --test_only \
               --proposals_num 1372 --square_size 4 --proposals_per_square 28  --workers 8 --lr 0.1 --svb --svb_factor 1.5  \
               --dataset flowers --print_freq 1 --arch vgg19_bn --num_part 3 --epochs_part 30 --num_classes 102 --num_select_proposals 50 \
               --epochs 160 --pretrained_model /home/lab-zhang.yabin/PAFGN/pytorch-PartNet/pretrain_vgg_imagenet/vgg19-bn-modified/model_best.pth.tar \

#python main.py --batch_size 128 --batch_size_partnet 64 --momentum 0.9 --weight_decay 1e-4 --data_path /data/CUB_200_2011/  --test_only \
#               --proposals_num 1372 --square_size 4 --proposals_per_square 28  --workers 8 --lr 0.1 --svb --svb_factor 1.5  --schedule 80 120 --schedule_part 10 20 \
#               --dataset cub200 --print_freq 1 --arch vgg19_bn --num_part 3 --epochs_part 30 --num_classes 200 --num_select_proposals 50 \
#               --epochs 160 --pretrained_model /home/lab-zhang.yabin/PAFGN/pytorch-PartNet/pretrain_vgg_imagenet/vgg19-bn-modified/model_best.pth.tar


########################################### use the following command to train the PartNet #######################################
#python main.py --batch_size 128 --batch_size_partnet 64 --momentum 0.9 --weight_decay 1e-4 --data_path /data/flower102/   \
#               --proposals_num 1372 --square_size 4 --proposals_per_square 28  --workers 8 --lr 0.1 --svb --svb_factor 1.5  \
#               --dataset flowers --print_freq 1 --arch vgg19_bn --num_part 3 --epochs_part 30 --num_classes 102 --num_select_proposals 50 \
#               --epochs 160 --pretrained_model /home/lab-zhang.yabin/PAFGN/pytorch-PartNet/pretrain_vgg_imagenet/vgg19-bn-modified/model_best.pth.tar \


#python main.py --batch_size 128 --batch_size_partnet 64 --momentum 0.9 --weight_decay 1e-4 --data_path /data/CUB_200_2011/   \
#               --proposals_num 1372 --square_size 4 --proposals_per_square 28  --workers 8 --lr 0.1 --svb --svb_factor 1.5  --schedule 80 120 --schedule_part 10 20 \
#               --dataset cub200 --print_freq 1 --arch vgg19_bn --num_part 3 --epochs_part 30 --num_classes 200 --num_select_proposals 50 \
#               --epochs 160 --pretrained_model /home/lab-zhang.yabin/PAFGN/pytorch-PartNet/pretrain_vgg_imagenet/vgg19-bn-modified/model_best.pth.tar

