import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.select_backbone import select_backbone

class AdaptationModel(nn.Module): 
    """
    Adaptation Model to adapt to the target domain
    """
    def __init__(self, args, device, network = 'i3d'):
        super(AdaptationModel, self).__init__()

        if args.modality == 'RGB' or args.modality == 'Joint':
            self.rgb_backbone = select_backbone(network, first_channel = 3)
            self.rgb_backbone.load_state_dict(torch.load(os.path.join(args.pretrained, 'rgb_imagenet.pt')), strict = True)
        
        if args.modality == 'Flow' or args.modality == 'Joint':
            self.flow_backbone = select_backbone(network, first_channel = 2)
            self.flow_backbone.load_state_dict(torch.load(os.path.join(args.pretrained, 'flow_imagenet.pt')), strict = True)
        
        if args.modality == 'RGB' or args.modality == 'Joint':
            self.rgb_backbone.replace_logits(args.num_classes)

        if args.modality == 'Flow' or args.modality == 'Joint':
            self.flow_backbone.replace_logits(args.num_classes)     

        
    def forward(self, video, args):
        
        if args.modality == 'RGB':
            rgb_logits, rgb_feat = self.rgb_backbone(video)

        elif args.modality == 'Flow':
            flow_logits, flow_feat = self.flow_backbone(video)
        
        elif args.modality == 'Joint':
            rgb_logits, rgb_feat = self.rgb_backbone(video[0])
            flow_logits, flow_feat = self.flow_backbone(video[1])

        if args.modality == 'RGB':
            return [rgb_logits], [rgb_feat]

        elif args.modality == 'Flow':
            return [flow_logits], [flow_feat]
        
        elif args.modality == 'Joint':
            return [rgb_logits, flow_logits], [rgb_feat, flow_feat]


class SourceOnlyModel(nn.Module):
    
    """
    The source-only model based on I3D
    """

    def __init__(self, args, network = 'i3d'):
        super(SourceOnlyModel, self).__init__()

        if args.modality == 'RGB' or args.modality == 'Joint':
            self.rgb_backbone = select_backbone(network, first_channel = 3)
            self.rgb_backbone.load_state_dict(torch.load(os.path.join(args.pretrained, 'rgb_imagenet.pt')), strict = True)
        
        if args.modality == 'Flow' or args.modality == 'Joint':
            self.flow_backbone = select_backbone(network, first_channel = 2)
            self.flow_backbone.load_state_dict(torch.load(os.path.join(args.pretrained, 'flow_imagenet.pt')), strict = True)
        
        if args.modality == 'RGB' or args.modality == 'Joint':
            self.rgb_backbone.replace_logits(args.num_classes)

        if args.modality == 'Flow' or args.modality == 'Joint':
            self.flow_backbone.replace_logits(args.num_classes)

    def forward(self, video, args):

        if args.modality == 'RGB':
            rgb_logits, rgb_feat  = self.rgb_backbone(video)

        elif args.modality == 'Flow':
            flow_logits, flow_feat = self.flow_backbone(video)
        
        elif args.modality == 'Joint':
            rgb_logits, rgb_feat = self.rgb_backbone(video[0])
            flow_logits, flow_feat = self.flow_backbone(video[1])

        if args.modality == 'RGB':
            return [rgb_logits], [rgb_feat]

        elif args.modality == 'Flow':
            return [flow_logits], [flow_feat]
        
        elif args.modality == 'Joint':
            return [rgb_logits, flow_logits], [rgb_feat, flow_feat]
