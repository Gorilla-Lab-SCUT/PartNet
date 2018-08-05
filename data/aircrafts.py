#################################################################
## The data preparation for the aircraft dataset
#################################################################
import os
import shutil
import torch
import torchvision.transforms as transforms
from data.folder_train import ImageFolder_train
from data.folder_download import ImageFolder_download

def split_train_test_images(data_dir):
    #data_dir = '/data/cars/'
    raise ValueError('the process of generate splited training and test images is empty')

def aircrafts(args, process_name, part_index):
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


