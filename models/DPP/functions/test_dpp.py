from .._ext import dpp
import torch
output = torch.FloatTensor(2744, 5).fill_(0)
histogram = torch.FloatTensor(28, 28).fill_(0)
score_sum = torch.FloatTensor(28, 28).fill_(0)
features = torch.randn(2, 512, 28, 28)
box_plan = torch.Tensor([[-1, -1, 1, 1],
                             [-2, -2, 2, 2],
                             [-1, -3, 1, 3],
                             [-3, -1, 3, 1],
                             [-3, -3, 3, 3],
                             [-2, -4, 2, 4],
                             [-4, -2, 4, 2],
                             [-4, -4, 4, 4],
                             [-3, -5, 3, 5],
                             [-5, -3, 5, 3],  # 10
                             [-5, -5, 5, 5],
                             [-4, -7, 4, 7],
                             [-7, -4, 7, 4],
                             [-6, -6, 6, 6],
                             [-4, -8, 4, 8],  # 15
                             [-8, -4, 8, 4],
                             [-7, -7, 7, 7],
                             [-5, -10, 5, 10],
                             [-10, -5, 10, 5],
                             [-8, -8, 8, 8],  # 20
                             [-6, -11, 6, 11],
                             [-11, -6, 11, 6],
                             [-9, -9, 9, 9],
                             [-7, -12, 7, 12],
                             [-12, -7, 12, 12],
                             [-10, -10, 10, 10],
                             [-7, -14, 7, 14],
                             [-14, -7, 14, 7]])
dpp.dpp_forward(4, 28, 1372, 16,
                box_plan, histogram, score_sum, output, features)

print(output)