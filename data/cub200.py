#################################################################
## The data preparation for the cub-200-2011 dataset
## The cub-200-2011 dataset can be downloaded at:
## http://www.vision.caltech.edu/visipedia/CUB-200.html
## All the downloaded related file should be placed at the "args.data_path".
#################################################################
import os
import shutil
import torch
import torchvision.transforms as transforms
from data.folder_train import ImageFolder_train
from data.folder_download import ImageFolder_download

def split_train_test_images(data_dir):
    #data_dir = '/home/lab-zhangyabin/project/fine-grained/CUB_200_2011/'
    src_dir = os.path.join(data_dir, 'images')
    target_dir = os.path.join(data_dir, 'splited_image')
    if not os.path.isdir(target_dir):
        os.makedirs(target_dir)
        print(src_dir)
    train_test_split = open(os.path.join(data_dir, 'train_test_split.txt'))
    line = train_test_split.readline()
    images = open(os.path.join(data_dir, 'images.txt'))
    images_line = images.readline()
    ##########################
    # print(images_line)
    image_list = str.split(images_line)
    # print(image_list[1])
    subclass_name = image_list[1].split('/')[0]
    # print(subclass_name)

    # print(line)
    class_list = str.split(line)[1]
    # print(class_list)

    print('begin to prepare the dataset CUB')
    count = 0
    while images_line:
        print(count)
        count = count + 1
        image_list = str.split(images_line)
        subclass_name = image_list[1].split('/')[0]  # get the name of the subclass
        # print(image_list[0])
        class_label = str.split(line)[1]  # get the label of the image
        # print(type(int(class_label)))
        test_or_train = 'train'
        if class_label == '0':  # the class belong to the train dataset
            test_or_train = 'val'
        train_test_dir = os.path.join(target_dir, test_or_train)
        if not os.path.isdir(train_test_dir):
            os.makedirs(train_test_dir)
        subclass_dir = os.path.join(train_test_dir, subclass_name)
        if not os.path.isdir(subclass_dir):
            os.makedirs(subclass_dir)

        souce_pos = os.path.join(src_dir, image_list[1])
        targer_pos = os.path.join(subclass_dir, image_list[1].split('/')[1])
        shutil.copyfile(souce_pos, targer_pos)
        images_line = images.readline()
        line = train_test_split.readline()

def cub200(args, process_name, part_index):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if process_name == 'image_classifier' or process_name == 'partnet' or process_name == 'download_proposals':
        traindir = os.path.join(args.data_path, 'splited_image/train')
        valdir = os.path.join(args.data_path, 'splited_image/val')
        if not os.path.isdir(traindir):
            print('the cub images are not well splited, split all images into train and val set')
            split_train_test_images(args.data_path)
        if process_name == 'download_proposals':
            print('the image pre-process for process: download proposals is Resize 512 and Center Crop 448')
            train_dataset = ImageFolder_download(
                root=traindir,
                transform=transforms.Compose([
                    transforms.Resize(512),
                    transforms.CenterCrop(448),
                    transforms.ToTensor(),
                    normalize,
                ]),
                transform_keep=transforms.Compose([
                    transforms.Resize(512),
                    transforms.CenterCrop(448),
                    transforms.ToTensor()
                ]),
                dataset_path=args.data_path
            )
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=args.batch_size_partnet, shuffle=False, num_workers=args.workers,
                pin_memory=True, sampler=None
            )
            val_loader = torch.utils.data.DataLoader(
                ImageFolder_download(
                    root=valdir,
                    transform=transforms.Compose([
                        transforms.Resize(512),
                        transforms.CenterCrop(448),
                        transforms.ToTensor(),
                        normalize,
                    ]),
                    transform_keep=transforms.Compose([
                        transforms.Resize(512),
                        transforms.CenterCrop(448),
                        transforms.ToTensor(),
                    ]),
                    dataset_path=args.data_path),
                batch_size=args.batch_size_partnet, shuffle=False,
                num_workers=args.workers, pin_memory=True
            )
            return train_loader, val_loader
        elif process_name == 'image_classifier':
            print('the image pre-process for process: image_classifier is Resize 512 and Random Crop 448')
            train_dataset = ImageFolder_train(
                traindir,
                transforms.Compose([
                    transforms.Resize(512),
                    transforms.RandomCrop(448),
                    transforms.ToTensor(),
                    normalize,
                ])
            )
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                pin_memory=True, sampler=None
            )
            val_loader = torch.utils.data.DataLoader(
                ImageFolder_train(valdir, transforms.Compose([
                    transforms.Resize(512),
                    transforms.CenterCrop(448),
                    transforms.ToTensor(),
                    normalize,
                ])),
                batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True
            )
            return train_loader, val_loader
        elif process_name == 'partnet':
            print('the image pre-process for process: partnet is Resize 512 and Random Crop 448')
            train_dataset = ImageFolder_train(
                traindir,
                transforms.Compose([
                    transforms.Resize(512),
                    transforms.RandomCrop(448),
                    transforms.ToTensor(),
                    normalize,
                ])
            )
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=args.batch_size_partnet, shuffle=True, num_workers=args.workers,
                pin_memory=True, sampler=None
            )
            val_loader = torch.utils.data.DataLoader(
                ImageFolder_train(valdir, transforms.Compose([
                    transforms.Resize(512),
                    transforms.CenterCrop(448),
                    transforms.ToTensor(),
                    normalize,
                ])),
                batch_size=args.batch_size_partnet, shuffle=False,
                num_workers=args.workers, pin_memory=True
            )
            return train_loader, val_loader
    elif process_name == 'part_classifiers':
        traindir = args.data_path + 'PartNet' + args.arch + '/part_' + str(part_index) + '/splited_image/train/'
        valdir = args.data_path + 'PartNet' + args.arch + '/part_' + str(part_index) + '/splited_image/val/'
        print('the image pre-process for process: part_classifier is Resize (448, 448) directly')
        train_dataset = ImageFolder_train(
            traindir,
            transforms.Compose([
                transforms.Resize((448, 448)),
                transforms.ToTensor(),
                normalize,
            ])
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
            pin_memory=True, sampler=None
        )
        val_loader = torch.utils.data.DataLoader(
            ImageFolder_train(valdir, transforms.Compose([
                transforms.Resize((448, 448)),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
            pin_memory=True, sampler=None
        )
        return train_loader, val_loader


