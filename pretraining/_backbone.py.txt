import timm
import torch.nn as nn

class OliveBackBone(nn.Module):
    def __init__(self, model_name='efficientvit_m5.r224_in1k', pretrained=True):
        super(OliveBackBone, self).__init__()
        self.model = timm.create_model(model_name, pretrained=True)
        self.model.head.linear = nn.Linear(self.model.head.linear.in_features, 600)

    def forward(self, x):
        x = self.model(x)
        return x
