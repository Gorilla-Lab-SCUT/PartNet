import argparse


def opts():
    parser = argparse.ArgumentParser(description='Train resnet on the cub dataset',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_path', type=str, default='/home/lab-zhangyabin/project/fine-grained/CUB_200_2011/',
                        help='Root of the data set')
    parser.add_argument('--dataset', type=str, default='cub200',
                        help='choose between flowers/cub200')
    # Optimization options
    parser.add_argument('--epochs', type=int, default=160, help='Number of epochs to train')
    parser.add_argument('--schedule', type=int, nargs='+', default=[80, 120],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--epochs_part', type=int, default=30, help='Number of epochs to train for the part classifier')
    parser.add_argument('--schedule_part', type=int, nargs='+', default=[10, 20],
                        help='Decrease learning rate at these epochs in the training of part classifier.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
    parser.add_argument('--batch_size_partnet', type=int, default=64, help='Batch size for partnet, set to 64 due to the GPU memory constrain')
    parser.add_argument('--lr', '--learning_rate', type=float, default=0.1, help='The initial learning rate.')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay (L2 penalty).')

    parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
    # checkpoints
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--resume', type=str, default='', help='Checkpoints path to resume(default none)')
    parser.add_argument('--pretrained_model', type=str, default='', help='Dir of the ImageNet pretrained modified models')
    parser.add_argument('--pretrain', action='store_true', help='whether using pretrained model')
    parser.add_argument('--test_only', action='store_true', help='Test only flag')
    # Architecture
    parser.add_argument('--arch', type=str, default='', help='Model name')
    parser.add_argument('--proposals_num', type=int, default=1372, help='the number of proposals for one image')
    parser.add_argument('--square_size', type=int, default=4, help='the side length of each cell in DPP module')
    parser.add_argument('--proposals_per_square', type=int, default=28, help='the num of proposals per square')
    parser.add_argument('--stride', type=int, default=16, help='Stride of the used model')
    parser.add_argument('--num_part', type=int, default=3, help='the number of part to be detected in the partnet')
    parser.add_argument('--num_classes', type=int, default=200, help='the number of fine-grained classes')
    parser.add_argument('--num_select_proposals', type=int, default=50, help='the number of fine-grained classes')

    parser.add_argument('--svb', action='store_true', help='whether apply svb on the classifier')
    parser.add_argument('--svb_factor', type=float, default=1.5, help='svb factor in the SVB method')
    # i/o
    parser.add_argument('--log', type=str, default='./checkpoints', help='Log folder')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--print_freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    args = parser.parse_args()
    return args
