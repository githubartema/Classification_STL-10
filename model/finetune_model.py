import os
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch

class NNetwork(torch.nn.Module):
    def __init__(self, base_model):
        super(NNetwork, self).__init__()
        self.backbone_features = torch.nn.Sequential(*list(base_model.children())[:-1])
        self.fc_in_features = list(base_model.children())[-1].in_features
            
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.80),
            torch.nn.Linear(in_features=self.fc_in_features, out_features=10),
            torch.nn.Softmax(dim=1))
    
    def forward(self, x):  
        x = self.backbone_features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
  
    def freeze_backbone(self):
        for _ in list(self.backbone_features.parameters()):
            _.requires_grad = False
        
        for module in list(self.backbone_features.modules()):
            if isinstance(module, torch.nn.BatchNorm2d):
                if hasattr(module, 'weight'):
                    module.weight.requires_grad_(False)
                if hasattr(module, 'bias'):
                    module.bias.requires_grad_(False)

    def unfreeze_layers_from(self, idx: int):
        for _ in list(self.backbone_features[idx:].parameters()):
            _.requires_grad = True
        
        for module in list(self.backbone_features[idx:].modules()):
            if isinstance(module, torch.nn.BatchNorm2d):
                if hasattr(module, 'weight'):
                    module.weight.requires_grad_(True)
                if hasattr(module, 'bias'):
                    module.bias.requires_grad_(True)  