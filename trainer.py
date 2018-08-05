import time
import torch
import os
import ipdb
from PIL import Image
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np

def train(train_loader, model, criterion, optimizer, epoch, log_now, process_name, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.train()
    adjust_learning_rate(optimizer, epoch, process_name, args)
    end = time.time()
    for i, (input, target, target_loss) in enumerate(train_loader):
        data_time.update(time.time() - end)
        if process_name == 'partnet':
            target = target.cuda(non_blocking=True)
            target_loss = target_loss.cuda(non_blocking=True)
            target_var = torch.autograd.Variable(target_loss)
        elif process_name == 'image_classifier' or process_name == 'part_classifiers':
            target = target.cuda(non_blocking=True)
            target_var = torch.autograd.Variable(target)
        else:
            raise ValueError('the required process type is not supported')
        input_var = torch.autograd.Variable(input)
        # print(target_var)
        # ipdb.set_trace()
        output = model(input_var)
        # print(output)
        loss = criterion(output, target_var)
        #mesure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        #compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
    log = open(os.path.join(log_now, 'log.txt'), 'a')
    log.write("\n")
    log.write("Train:epoch: %d, loss: %4f, Top1 acc: %3f, Top5 acc: %3f" % (epoch, losses.avg, top1.avg, top5.avg))
    log.close()


def validate(val_loader, model, criterion, epoch, log_now, process_name, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target, target_loss) in enumerate(val_loader):
        target = target.cuda(non_blocking=True)
        input_var = torch.autograd.Variable(input)
        if process_name == 'partnet':
            target = target.cuda(non_blocking=True)
            target_loss = target_loss.cuda(non_blocking=True)
            target_var = torch.autograd.Variable(target_loss)
        elif process_name == 'image_classifier' or process_name == 'part_classifiers':
            target = target.cuda(non_blocking=True)
            target_var = torch.autograd.Variable(target)
        # compute output
        with torch.no_grad():
            output = model(input_var)
        loss = criterion(output, target_var)
        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))
    log = open(os.path.join(log_now, 'log.txt'), 'a')
    log.write("\n")
    log.write("                               Train:epoch: %d, loss: %4f, Top1 acc: %3f, Top5 acc: %3f" %\
              (epoch, losses.avg, top1.avg, top5.avg))
    log.close()
    return top1.avg

def download_part_proposals(data_loader, model, epoch, log_now, process_name, train_or_val, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    top1 = AverageMeter()
    top5 = AverageMeter()
    model.eval()
    # adjust_learning_rate(optimizer, epoch, args)
    end = time.time()
    tensor2image = transforms.ToPILImage()
    if train_or_val == 'train':
        num_selected = args.num_select_proposals
    elif train_or_val == 'val':
        num_selected = 1
    else:
        raise ValueError('only accept train or val module')
    for i, (input, input_keep, target, target_loss, path_image) in enumerate(data_loader):
        data_time.update(time.time() - end)
        target = target.cuda(non_blocking=True)
        target_loss = target_loss.cuda(non_blocking=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target_loss)

        proposals_scores_h = torch.Tensor(input.size(0), args.proposals_num, args.num_part + 1).fill_(
            0)  # all the proposals scores
        all_proposals = torch.Tensor(input.size(0) * args.proposals_num, 5).fill_(0)  # all the

        def hook_scores(module, inputdata, output):
            proposals_scores_h.copy_(output.data)

        def hook_proposals(module, inputdata, output):
            all_proposals.copy_(output.data)

        handle_scores = model.module.detection_stream.softmax_cls.register_forward_hook(hook_scores)
        handle_proposals = model.module.DPP.register_forward_hook(hook_proposals)
        with torch.no_grad():
            output = model(input_var)

        handle_proposals.remove()  ## delete the hook after used.
        handle_scores.remove()
        # print(output)
        for j in range(input.size(0)):
            real_image = tensor2image(input_keep[j])
            scores_for_image = proposals_scores_h[j]
            value, sort = torch.sort(scores_for_image, 0, descending=True)  # the score from large to small
            # print('value:', value)
            # print('sort:', sort)
            proposals_one = all_proposals[j*args.proposals_num:(j+1)*args.proposals_num, 1:5]
            # print(sort)
            #check whether the dir file exist, if not, create one.
            img_dir = path_image[j]

            # ipdb.set_trace()
            for num_p in range(0, args.num_part+1):
                last_dash = img_dir.rfind('/')
                image_folder = args.data_path + 'PartNet' + args.arch + '/part_' + str(num_p) + '/' + img_dir[0:last_dash]
                if not os.path.isdir(image_folder):
                    os.makedirs(image_folder)
                image_name = img_dir[last_dash:]
                last_point = image_name.find('.')
                name = image_name[0:last_point]
                one_part_sort = sort[:, num_p]
                for k in range(num_selected): # for each proposals
                    select_proposals = np.array(proposals_one[one_part_sort[k]])
                    cropped_image = real_image.crop((select_proposals[0], select_proposals[1], select_proposals[2], select_proposals[3]))
                    dir_to_save = image_folder + name + '_' + str(k) + image_name[last_point:]
                    cropped_image.save(dir_to_save)

        #mesure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1,5))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            print(train_or_val + ': [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(data_loader), batch_time=batch_time,
                   data_time=data_time, top1=top1, top5=top5))
    log = open(os.path.join(log_now, 'log.txt'), 'a')
    log.write("\n")
    log.write(train_or_val + ":epoch: %d, Top1 acc: %3f, Top5 acc: %3f" % (epoch, top1.avg, top5.avg))
    log.close()
    return top1.avg

def download_scores(val_loader, model, log_now, process_name, args):
    if not os.path.isdir(log_now):
        raise ValueError('the log dir request is not exist')
    file_to_save_or_load = log_now + '/' + process_name + '.pth.tar'
    probabilities = []
    labels = []
    batch_time = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # switch to evaluate mode
    model.eval()
    end = time.time()
    softmax = nn.Softmax()
    for i, (input, target, target_loss) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)

        # compute output
        with torch.no_grad():
            output = model(input_var)
            # print(output)
            if process_name == 'partnet':
                output = torch.nn.functional.normalize(output, p=1, dim=1)
            else:
                output = softmax(output)

        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        for j in range(input.size(0)):
            # maybe here need to sub tensor to save memory.
            probabilities.append(output.data[j].cpu().clone())
            labels.append(target[j])

        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        print('Test: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
              'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
            1, i, len(val_loader), batch_time=batch_time,
            top1=top1, top5=top5))

    log = open(os.path.join(log_now, 'log.txt'), 'a')
    log.write("\n")
    log.write(process_name)
    log.write("     Val:epoch: %d, Top1 acc: %3f, Top5 acc: %3f" % \
              (1, top1.avg, top5.avg))
    log.close()
    torch.save({'scores': probabilities, 'labels': labels}, file_to_save_or_load)

    return probabilities, labels

def svb(model, args):
    print('the layer used for svb is', model.module.classification_stream.classifier[3])
    svb_model = model.module.classification_stream.classifier[3]
    tmpbatchM = svb_model.weight.data.t().clone()
    tmpU, tmpS, tmpV = torch.svd(tmpbatchM)
    for idx in range(0, tmpS.size(0)):
        if tmpS[idx] > args.svb_factor:
            tmpS[idx] = args.svb_factor
        elif tmpS[idx] < 1 / args.svb_factor:
            tmpS[idx] = 1 / args.svb_factor
    tmpbatchM = torch.mm(torch.mm(tmpU, torch.diag(tmpS.cuda())), tmpV.t()).t().contiguous()
    svb_model.weight.data.copy_(tmpbatchM.view_as(svb_model.weight.data))

def svb_det(model, args): ## it is not use in our experiments
    print('the layer used for svb is', model.module.detection_stream.detector[3])
    svb_model = model.module.detection_stream.detector[3]
    tmpbatchM = svb_model.weight.data.t().clone()
    tmpU, tmpS, tmpV = torch.svd(tmpbatchM)
    for idx in range(0, tmpS.size(0)):
        if tmpS[idx] > args.svb_factor:
            tmpS[idx] = args.svb_factor
        elif tmpS[idx] < 1 / args.svb_factor:
            tmpS[idx] = 1 / args.svb_factor
    tmpbatchM = torch.mm(torch.mm(tmpU, torch.diag(tmpS.cuda())), tmpV.t()).t().contiguous()
    svb_model.weight.data.copy_(tmpbatchM.view_as(svb_model.weight.data))

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, process_name, args):
    """Adjust the learning rate according the epoch"""
    # print(epoch)
    # print(args.schedule[1])
    if process_name == 'part_classifiers':
        exp = epoch >= args.schedule_part[1] and 2 or epoch >= args.schedule_part[0] and 1 or 0
        exp_pre = epoch >= args.schedule_part[1] and 2 or epoch >= args.schedule_part[0] and 2 or 2
    elif process_name == 'partnet':
        exp = epoch >= args.schedule[1] and 2 or epoch >= args.schedule[0] and 1 or 0
        exp_pre = epoch >= args.schedule[1] and 2 or epoch >= args.schedule[0] and 2 or 2
    else:
        exp = epoch >= args.schedule[1] and 2 or epoch >= args.schedule[0] and 1 or 0
        exp_pre = epoch >= args.schedule[1] and 2 or epoch >= args.schedule[0] and 2 or 2
   
    # print(exp)
    lr = args.lr * (args.gamma ** exp)
    lr_pre = args.lr * (args.gamma ** exp_pre)
    print('LR for new-added', lr)
    print('LR for old', lr_pre)
    for param_group in optimizer.param_groups:
        if param_group['name'] == 'pre-trained':
            param_group['lr'] = lr_pre
        else:
            param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
