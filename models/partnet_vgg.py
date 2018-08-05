##################################################
## The model construction of the PartNet.
##################################################

import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math
import torch
from models.roi_pooling.modules.roi_pool import _RoIPooling  ## Roi_Pooling Function
from models.DPP.modules.dpp import _DPP  ## DPP function to generate part proposals from the feature maps directly.
from torch.autograd import Function
from torch.nn.modules.module import Module
import time
import copy
import ipdb


__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000):
        super(VGG, self).__init__()
        self.features = features
        # self.classifier = nn.Sequential(
        #     nn.Linear(512 * 7 * 7, 4096),
        #     nn.ReLU(True),
        #     nn.Dropout(),
        #     nn.Linear(4096, 4096),
        #     nn.ReLU(True),
        #     nn.Dropout(),
        #     nn.Linear(4096, num_classes),
        # )
        #self.avgpool = nn.AvgPool2d(28)
        # self.conv = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        # x = self.conv(x)
        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    print('the vgg network structure has been changed, the last max pooling layer has been dropped.')
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
}


def vgg11(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['A']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg11']))
    return model


def vgg11_bn(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['A'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg11_bn']))
    return model


def vgg13(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['B']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg13']))
    return model


def vgg13_bn(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['B'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg13_bn']))
    return model


def vgg16(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['D']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16']))
    return model

# class DPPFunction(Function):
#     def __init__(self, square_size, proposals_per_square, proposals_per_image, spatial_scale):
#         self.square_size = square_size
#         self.proposals_per_square = proposals_per_square
#         self.spatital_scale = spatial_scale
#         self.output = torch.cuda.FloatTensor()
#         self.proposals_per_image = proposals_per_image
#         self.box_plan = torch.Tensor([[-1, -1, 1, 1],
#                                       [-2, -2, 2, 2],
#                                       [-1, -3, 1, 3],
#                                       [-3, -1, 3, 1],
#                                       [-3, -3, 3, 3],
#                                       [-2, -4, 2, 4],
#                                       [-4, -2, 4, 2],
#                                       [-4, -4, 4, 4],
#                                       [-3, -5, 3, 5],
#                                       [-5, -3, 5, 3], # 10
#                                       [-5, -5, 5, 5],
#                                       [-4, -7, 4, 7],
#                                       [-7, -4, 7, 4],
#                                       [-6, -6, 6, 6],
#                                       [-4, -8, 4, 8], # 15
#                                       [-8, -4, 8, 4],
#                                       [-7, -7, 7, 7],
#                                       [-5, -10, 5, 10],
#                                       [-10, -5, 10, 5],
#                                       [-8, -8, 8, 8],  # 20
#                                       [-6, -11, 6, 11],
#                                       [-11, -6, 11, 6],
#                                       [-9, -9, 9, 9],
#                                       [-7, -12, 7, 12],
#                                       [-12, -7, 12, 12],
#                                       [-10, -10, 10, 10],
#                                       [-7, -14, 7, 14],
#                                       [-14, -7, 14, 7]])
#         self.square_num = int(28/self.square_size)  ### it should be 7
#         calculate_num = self.square_num * self.square_num * proposals_per_square
#         if self.proposals_per_image != calculate_num:
#             raise ValueError('the number generated by dpp should be', calculate_num)
#         if self.square_size != 4 and self.square_size != 7:
#             raise ValueError('the number of the square for one line should be 4 or 7, but you define:', self.square_size)
#         if self.proposals_per_square > 28:
#             raise ValueError('the proposals number for each image should below 28')
#
#
#     def forward(self, features):
#         timer = time.time()
#         batch_size, num_channels, data_height, data_width = features.size()
#         features_float = features.float()
#
#         num_rois = batch_size * self.proposals_per_image
#         output = torch.Tensor(num_rois, 5).fill_(0)
#
#         if self.spatital_scale != 16:
#             raise ValueError('the spatial scale should be 16, but you define is:', self.spatital_scale)
#         proposals_index = 0
#
#         for i in range(batch_size):
#             roi_memory = torch.Tensor(28,28).zero_()
#             roi_score = torch.sum(features_float[i], 0)
#             output[i*self.proposals_per_image: (i+1)*self.proposals_per_image, 0] = i
#             for j in range(num_channels):
#                 one_channel = features_float[i][j]
#                 max_per_row, max_column_per_row = torch.max(one_channel, 1)
#                 max_p, max_row = torch.max(max_per_row, 0)
#                 x_center = max_row[0]
#                 max_col = max_column_per_row[x_center]
#                 y_center = max_col
#                 roi_memory[x_center][y_center] = roi_memory[x_center][y_center] + 1
#             for x in range(self.square_num):
#                 for y in range(self.square_num):
#                     temp = roi_memory[x * self.square_size:(x+1)*self.square_size, y*self.square_size: (y+1)*self.square_size]
#                     max_per_row, max_column_per_row = torch.max(temp, 1)
#                     max_p, max_row = torch.max(max_per_row, 0)
#                     x_center = max_row[0]
#                     max_col = max_column_per_row[x_center]
#                     y_center = max_col
#                     find_repeat = torch.eq(temp, temp[x_center][y_center])
#                     if torch.sum(find_repeat) > 1:
#                         score_temp = roi_score[x * self.square_size:(x+1)*self.square_size, y*self.square_size:(y+1)*self.square_size]
#                         max_per_row, max_column_per_row = torch.max(score_temp, 1)
#                         max_p, max_row = torch.max(max_per_row, 0)
#                         x_center = max_row[0]
#                         max_col = max_column_per_row[x_center]
#                         y_center = max_col
#                     x_center = x_center + x * self.square_size
#                     y_center = y_center + y * self.square_size
#                     for k in range(self.proposals_per_square):
#                         output[proposals_index][1:5] = torch.Tensor([x_center +self.box_plan[k][0], y_center+self.box_plan[k][1], x_center+self.box_plan[k][2], y_center+self.box_plan[k][3]])
#                         proposals_index = proposals_index + 1
#         output[:, 1:5].mul_(self.spatital_scale)
#         output[:, 1:5].clamp_(0, 448)
#         #self.output.resize_(output.size()).copy_(output, non_blocking=False)
#         print('the dpp time is:', time.time() - timer)
#         return output.cuda()
#
#
#     def backward(self, grad_output):
#
#         return None

# class DPP(Module):
#     def __init__(self, square_size, proposals_per_square, proposals_per_image, spatial_scale):
#         super(DPP, self).__init__()
#
#         self.square_size = int(square_size)
#         self.proposals_per_square = int(proposals_per_square)
#         self.proposals_per_image = int(proposals_per_image)
#         self.spatial_scale = int(spatial_scale)
#
#     def forward(self, features):
#         return DPPFunction(self.square_size, self.proposals_per_square, self.proposals_per_image, self.spatial_scale)(features)

class Classification_stream(Module):
    def __init__(self, proposal_num, num_classes):
        super(Classification_stream, self).__init__()
        self.proposals_num = proposal_num
        self.num_classes = num_classes
        self.classifier = nn.Sequential(
            nn.Linear(512 * 3 * 3, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, self.num_classes+1),
        )
        self.softmax = nn.Softmax(2)
    def forward(self, x):
        x = self.classifier(x)
        x = x.view(-1, self.proposals_num, self.num_classes+1)
        x = self.softmax(x)
        x = x.narrow(2, 0, self.num_classes)
        return x

class Detection_stream(Module):
    def __init__(self, proposal_num, part_num):
        super(Detection_stream, self).__init__()
        self.proposals_num = proposal_num
        self.part_num = part_num
        self.detector = nn.Sequential(
            nn.Linear(512*3*3, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, self.part_num+1),
        )
        self.softmax_cls = nn.Softmax(2)
        self.softmax_nor = nn.Softmax(1)
    def forward(self, x):
        x = self.detector(x)
        x = x.view(-1, self.proposals_num, self.part_num+1)
        x = self.softmax_cls(x)
        x = x.narrow(2, 0, self.part_num)
        x = self.softmax_nor(x)
        return x

class construct_partnet(nn.Module):
    def __init__(self, conv_model, args):
        super(construct_partnet, self).__init__()
        self.conv_model = conv_model
        self.roi_pool = _RoIPooling(3, 3, 1.0/16)
        self.args = args
        self.DPP = _DPP(args.square_size, args.proposals_per_square, args.proposals_num, args.stride)
        self.classification_stream = Classification_stream(args.proposals_num, args.num_classes)
        self.detection_stream = Detection_stream(args.proposals_num, args.num_part)

    def forward(self, x):
        x = self.conv_model(x)
        rois = self.DPP(x)
        x = self.roi_pool(x, rois)
        x = x.view(x.size(0), -1)
        x_c = self.classification_stream(x)
        x_d = self.detection_stream(x)
        x_c = x_c.transpose(1, 2)
        mix = torch.matmul(x_c, x_d)
        x = mix.mean(2)

        return x

def vgg16_bn(args, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)
    if args.pretrain:
        print('load the imageNet pretrained model')
        pretrained_dict = model_zoo.load_url(model_urls['vgg16_bn'])
        model_dict = model.state_dict()
        pretrained_dict_temp = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict_temp)
        model.load_state_dict(model_dict)
    print(args.finetuned_model)
    if args.finetuned_model != '':
        print('load the model that has been finetuned on cub', args.finetuned_model)
        pretrained_dict = torch.load(args.finetuned_model)['state_dict']
        pretrained_dict_temp = copy.deepcopy(pretrained_dict)
        model_state_dict = model.state_dict()

        for k_tmp in pretrained_dict_temp.keys():
            if k_tmp.find('module.base_conv') != -1:
                k = k_tmp.replace('module.base_conv.', '')
                pretrained_dict[k] = pretrained_dict.pop(k_tmp)
        # ipdb.set_trace()
        # print(pretrained_dict)
        pretrained_dict_temp2 = {k: v for k, v in pretrained_dict.items() if k in model_state_dict}
        # print(pretrained_dict_temp2)
        model_state_dict.update(pretrained_dict_temp2)
        model.load_state_dict(model_state_dict)

        print('here load the fine tuned conv layer to the model')

    ### change the model to PartNet by ADD: DPP + ROIPooling + two stream.
    PartNet = construct_partnet(model, args)

    return PartNet


def vgg19(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration "E")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['E']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19']))
    return model




def vgg19_bn(args, **kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['E'], batch_norm=True), **kwargs)
    finetuned_model = args.dataset + '/Image_Classifier/model_best.pth.tar'
    if args.pretrain:
        print('load the imageNet pretrained model')
        pretrained_dict = model_zoo.load_url(model_urls['vgg19_bn'])
        model_dict = model.state_dict()
        pretrained_dict_temp = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict_temp)
        model.load_state_dict(model_dict)
    if finetuned_model != '':
        print('load the model that has been finetuned on cub', finetuned_model)
        pretrained_dict = torch.load(finetuned_model)['state_dict']
        pretrained_dict_temp = copy.deepcopy(pretrained_dict)
        model_state_dict = model.state_dict()

        for k_tmp in pretrained_dict_temp.keys():
            if k_tmp.find('module.base_conv') != -1:
                k = k_tmp.replace('module.base_conv.', '')
                pretrained_dict[k] = pretrained_dict.pop(k_tmp)
        pretrained_dict_temp2 = {k: v for k, v in pretrained_dict.items() if k in model_state_dict}
        model_state_dict.update(pretrained_dict_temp2)
        model.load_state_dict(model_state_dict)
    PartNet = construct_partnet(model, args)
    return PartNet


def partnet_vgg(args, **kwargs):
    print("==> creating model '{}' ".format(args.arch))
    if args.arch == 'vgg19_bn':
        return vgg19_bn(args)
    elif args.arch == 'vgg16_bn':
        return vgg16_bn(args)
    else:
        raise ValueError('Unrecognized model architecture', args.arch)
