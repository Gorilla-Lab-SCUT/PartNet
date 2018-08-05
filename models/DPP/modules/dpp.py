from torch.nn.modules.module import Module
from ..functions.dpp import DPPFunction


class _DPP(Module):
    def __init__(self, square_size, proposals_per_square, proposals_per_image, spatial_scale):
        super(_DPP, self).__init__()

        self.square_size = int(square_size)
        self.proposals_per_square = int(proposals_per_square)
        self.proposals_per_image = int(proposals_per_image)
        self.spatial_scale = int(spatial_scale)

    def forward(self, features):
        return DPPFunction(self.square_size, self.proposals_per_square, self.proposals_per_image, self.spatial_scale)(features)
