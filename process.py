import os
import json
import shutil
import torch.optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
from models.model_construct import Model_Construct
from data.prepare_data import generate_dataloader
from trainer import train  # For the training process
from trainer import validate  # For the validate (test) process
from trainer import download_part_proposals
from trainer import download_scores
from trainer import svb
from trainer import svb_det
import time

def Process1_Image_Classifier(args):

    log_now = args.dataset + '/Image_Classifier'
    process_name = 'image_classifier'
    if os.path.isfile(log_now + '/final.txt'):
        print('the Process1_Image_Classifier is finished')
        return
    best_prec1 = 0
    model = Model_Construct(args, process_name)
    model = torch.nn.DataParallel(model).cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD([
        {'params': model.module.base_conv.parameters(), 'name': 'pre-trained'},
        {'params': model.module.fc.parameters(), 'lr': args.lr, 'name': 'new-added'}
    ],
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    start_epoch = args.start_epoch
    if args.resume:
        if os.path.isfile(args.resume):
            print("==> loading checkpoints '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("==> loaded checkpoint '{}'(epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            args.resume = ''
        else:
            raise ValueError('The file to be resumed from is not exited', args.resume)
    else:
        if not os.path.isdir(log_now):
            os.makedirs(log_now)
        log = open(os.path.join(log_now, 'log.txt'), 'w')
        state = {k: v for k, v in args._get_kwargs()}
        log.write(json.dumps(state) + '\n')
        log.close()
    cudnn.benchmark = True
    train_loader, val_loader = generate_dataloader(args, process_name, -1)
    if args.test_only:
        validate(val_loader, model, criterion, 2000, args)

    for epoch in range(start_epoch, args.epochs):
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, log_now, process_name, args)
        # evaluate on the val data
        prec1 = validate(val_loader, model, criterion, epoch, log_now, process_name, args)
        # record the best prec1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        if is_best:
            log = open(os.path.join(log_now, 'log.txt'), 'a')
            log.write(
                "best acc %3f" % (best_prec1))
            log.close()
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best, log_now)
    #download_scores(val_loader, model, log_now, process_name, args)
    log = open(os.path.join(log_now, 'final.txt'), 'w')
    log.write(
        "best acc %3f" % (best_prec1))
    log.close()


def Process2_PartNet(args):
    log_now = args.dataset + '/PartNet'
    process_name = 'partnet'
    if os.path.isfile(log_now + '/final.txt'):
        print('the Process2_PartNet is finished')
        return
    best_prec1 = 0
    model = Model_Construct(args, process_name)
    model = torch.nn.DataParallel(model).cuda()
    criterion = nn.BCELoss().cuda()
    # print(model)
    # print('the learning rate for the new added layer is set to 1e-3 to slow down the speed of learning.')
    optimizer = torch.optim.SGD([
        {'params': model.module.conv_model.parameters(), 'name': 'pre-trained'},
        {'params': model.module.classification_stream.parameters(), 'name': 'new-added'},
        {'params': model.module.detection_stream.parameters(), 'name': 'new-added'}
    ],
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay)
    start_epoch = args.start_epoch
    if args.resume:
        if os.path.isfile(args.resume):
            print("==> loading checkpoints '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("==> loaded checkpoint '{}'(epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            args.resume = ''
        else:
            raise ValueError('The file to be resumed from is not exited', args.resume)
    else:
        if not os.path.isdir(log_now):
            os.makedirs(log_now)
        log = open(os.path.join(log_now, 'log.txt'), 'w')
        state = {k: v for k, v in args._get_kwargs()}
        log.write(json.dumps(state) + '\n')
        log.close()
    cudnn.benchmark = True
    train_loader, val_loader = generate_dataloader(args, process_name, -1)
    if args.test_only:
        validate(val_loader, model, criterion, 2000, args)
    for epoch in range(start_epoch, args.epochs):
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, log_now, process_name, args)
        # evaluate on the val data
        prec1 = validate(val_loader, model, criterion, epoch, log_now, process_name, args)
        # record the best prec1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        if is_best:
            log = open(os.path.join(log_now, 'log.txt'), 'a')
            log.write(
                "best acc %3f" % (best_prec1))
            log.close()
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best, log_now)
        svb_timer = time.time()
        if args.svb and epoch != (args.epochs - 1):
            svb(model, args)
            print('!!!!!!!!!!!!!!!!!! the svb constrain is only applied on the classification stream.')
            svb_det(model, args)
            print('the svb time is: ', time.time() - svb_timer)
    #download_scores(val_loader, model, log_now, process_name, args)
    log = open(os.path.join(log_now, 'final.txt'), 'w')
    log.write(
        "best acc %3f" % (best_prec1))
    log.close()

def Process3_Download_Proposals(args):
    log_now = args.dataset + '/Download_Proposals'
    process_name = 'download_proposals'
    if os.path.isfile(log_now + '/final.txt'):
        print('the Process3_download proposals is finished')
        return

    model = Model_Construct(args, process_name)
    model = torch.nn.DataParallel(model).cuda()

    optimizer = torch.optim.SGD([
        {'params': model.module.conv_model.parameters(), 'name': 'pre-trained'},
        {'params': model.module.classification_stream.parameters(), 'name': 'new-added'},
        {'params': model.module.detection_stream.parameters(), 'name': 'new-added'}
    ],
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay)
    log_partnet_model = args.dataset + '/PartNet/model_best.pth.tar'
    checkpoint = torch.load(log_partnet_model)
    model.load_state_dict(checkpoint['state_dict'])
    print('load the pre-trained partnet model from:', log_partnet_model)

    if args.resume:
        if os.path.isfile(args.resume):
            print("==> loading checkpoints '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("==> loaded checkpoint '{}'(epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            args.resume = ''
        else:
            raise ValueError('The file to be resumed from is not exited', args.resume)
    else:
        if not os.path.isdir(log_now):
            os.makedirs(log_now)
        log = open(os.path.join(log_now, 'log.txt'), 'w')
        state = {k: v for k, v in args._get_kwargs()}
        log.write(json.dumps(state) + '\n')
        log.close()

    cudnn.benchmark = True
    train_loader, val_loader = generate_dataloader(args, process_name)

    for epoch in range(1):

        download_part_proposals(train_loader, model, epoch, log_now, process_name, 'train', args)

        best_prec1 = download_part_proposals(val_loader, model, epoch, log_now, process_name, 'val', args)

    log = open(os.path.join(log_now, 'final.txt'), 'w')
    log.write(
        "best acc %3f" % (best_prec1))
    log.close()


def Process4_Part_Classifiers(args):
    for i in range(args.num_part):  ### if the process is break in this section, more modification is needed.
        log_now = args.dataset + '/Part_Classifiers_' + str(i)
        process_name = 'part_classifiers'
        if os.path.isfile(log_now + '/final.txt'):
            print('the Process4_Part_Classifier is finished', i)
            continue
        best_prec1 = 0
        model = Model_Construct(args, process_name)
        model = torch.nn.DataParallel(model).cuda()
        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = torch.optim.SGD([
            {'params': model.module.base_conv.parameters(), 'name': 'pre-trained'},
            {'params': model.module.fc.parameters(), 'lr': args.lr, 'name': 'new-added'}
        ],
                                    lr=args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        log_image_model = args.dataset + '/Image_Classifier/model_best.pth.tar'
        checkpoint = torch.load(log_image_model)
        model.load_state_dict(checkpoint['state_dict'])
        print('load the cub fine-tuned model from:', log_image_model)
        start_epoch = args.start_epoch
        if args.resume:
            if os.path.isfile(args.resume):
                print("==> loading checkpoints '{}'".format(args.resume))
                checkpoint = torch.load(args.resume)
                start_epoch = checkpoint['epoch']
                best_prec1 = checkpoint['best_prec1']
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                print("==> loaded checkpoint '{}'(epoch {})"
                      .format(args.resume, checkpoint['epoch']))
                args.resume = ''
            else:
                raise ValueError('The file to be resumed from is not exited', args.resume)
        else:
            if not os.path.isdir(log_now):
                os.makedirs(log_now)
            log = open(os.path.join(log_now, 'log.txt'), 'w')
            state = {k: v for k, v in args._get_kwargs()}
            log.write(json.dumps(state) + '\n')
            log.close()
        cudnn.benchmark = True
        train_loader, val_loader = generate_dataloader(args, process_name, i)
        if args.test_only:
            validate(val_loader, model, criterion, 2000, args)
        for epoch in range(start_epoch, args.epochs_part):
            # train for one epoch
            train(train_loader, model, criterion, optimizer, epoch, log_now, process_name, args)
            # evaluate on the val data
            prec1 = validate(val_loader, model, criterion, epoch, log_now, process_name, args)
            # record the best prec1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            if is_best:
                log = open(os.path.join(log_now, 'log.txt'), 'a')
                log.write(
                    "best acc %3f" % (best_prec1))
                log.close()
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, log_now)
        #download_scores(val_loader, model, log_now, process_name, args)
        log = open(os.path.join(log_now, 'final.txt'), 'w')
        log.write(
            "best acc %3f" % (best_prec1))
        log.close()

def Process5_Final_Result(args):
    ############################# Image Level Classifier #############################
    log_now = args.dataset + '/Image_Classifier'
    process_name = 'image_classifier'
    model = Model_Construct(args, process_name)
    model = torch.nn.DataParallel(model).cuda()
    pre_trained_model = log_now + '/model_best.pth.tar'
    checkpoint = torch.load(pre_trained_model)
    model.load_state_dict(checkpoint['state_dict'])
    train_loader, val_loader = generate_dataloader(args, process_name, -1)
    download_scores(val_loader, model, log_now, process_name, args)
    ############################# PartNet ############################################
    log_now = args.dataset + '/PartNet'
    process_name = 'partnet'
    model = Model_Construct(args, process_name)
    model = torch.nn.DataParallel(model).cuda()
    pre_trained_model = log_now + '/model_best.pth.tar'
    checkpoint = torch.load(pre_trained_model)
    model.load_state_dict(checkpoint['state_dict'])
    train_loader, val_loader = generate_dataloader(args, process_name)
    download_scores(val_loader, model, log_now, process_name, args)
    ############################# Three Part Level Classifiers #######################
    for i in range(args.num_part):  ### if the process is break in this section, more modification is needed.
        log_now = args.dataset + '/Part_Classifiers_' + str(i)
        process_name = 'part_classifiers'
        model = Model_Construct(args, process_name)
        model = torch.nn.DataParallel(model).cuda()
        pre_trained_model = log_now + '/model_best.pth.tar'
        checkpoint = torch.load(pre_trained_model)
        model.load_state_dict(checkpoint['state_dict'])
        train_loader, val_loader = generate_dataloader(args, process_name, i)
        download_scores(val_loader, model, log_now, process_name, args)


    log_image = args.dataset + '/Image_Classifier'
    process_image = 'image_classifier'

    log_partnet = args.dataset + '/PartNet'
    process_partnet = 'partnet'

    log_part0 = args.dataset + '/Part_Classifiers_' + str(0)
    process_part0 = 'part_classifiers'

    log_part1 = args.dataset + '/Part_Classifiers_' + str(1)
    process_part1 = 'part_classifiers'

    log_part2 = args.dataset + '/Part_Classifiers_' + str(2)
    process_part2 = 'part_classifiers'

    image_table = torch.load(log_image + '/' + process_image + '.pth.tar')
    image_probability = image_table['scores']
    labels = image_table['labels']
    partnet_table = torch.load(log_partnet + '/' + process_partnet + '.pth.tar')
    partnet_probability = partnet_table['scores']
    #######################
    part0_table = torch.load(log_part0 + '/' + process_part0 + '.pth.tar')
    part0_probability = part0_table['scores']
    ##########################
    part1_table = torch.load(log_part1 + '/' + process_part1 + '.pth.tar')
    part1_probability = part1_table['scores']
    ##########################
    part2_table = torch.load(log_part2 + '/' + process_part2 + '.pth.tar')
    part2_probability = part2_table['scores']
    ##########################

    probabilities_group = []
    probabilities_group.append(image_probability)
    probabilities_group.append(part0_probability)
    probabilities_group.append(part1_probability)
    probabilities_group.append(part2_probability)
    probabilities_group.append(partnet_probability)
    count = 0
    for i in range(len(labels)):
        probability = probabilities_group[0][i]
        for j in range(len(probabilities_group)):
            probability = probabilities_group[j][i] + probability
        probability = probability - probabilities_group[0][i]
        label = labels[i]
        value, index = probability.sort(0, descending=True)
        if index[0] == label:
            count = count + 1
    top1 = count / len(labels)
    print('the final results obtained by averaging part0-1-2 image partnet is', top1)

def save_checkpoint(state, is_best, log_now):
    filename = 'checkpoint.pth.tar'
    dir_save_file = os.path.join(log_now, filename)
    torch.save(state, dir_save_file)
    if is_best:
        shutil.copyfile(dir_save_file, os.path.join(log_now, 'model_best.pth.tar'))
