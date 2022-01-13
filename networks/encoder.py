import torchvision.models as models
import torch
from torch import nn
import numpy as np

class encoder(nn.Module):
    def __init__(self,fix_weight=False, encoder_type = 'resnet101', pretrained=True, is_normalized=True):
        '''
        fix_weight: bool, decide the encoder is trainable or not
        encoder_type: str, the architecture of encoder
        pretrained: bool, decide if load weight that pretrained on ImageNet
        is_normalized: bool, decide if apply normalization to the input image
        '''
        super().__init__()
        if encoder_type == 'resnet101':
            self.resnet = models.resnet101(pretrained=pretrained)
        elif encoder_type == 'resnet18':
            self.resnet = models.resnet18(pretrained=pretrained)
        else:
            print('error: No such model')
            exit(0)
        self.layer1 = nn.Sequential(*list(self.resnet.children())[0:5]) #256, 56, 56
        self.layer2 = nn.Sequential(*list(self.resnet.children())[5]) #512 , 28, 28
        self.layer3 = nn.Sequential(*list(self.resnet.children())[6]) #1024 , 14, 14
        self.layer4 = nn.Sequential(*list(self.resnet.children())[7]) #2048 , 7, 7
        self.fix_resnet(fix_weight)
        self.mean = torch.from_numpy(np.array([0.485, 0.456, 0.406]).astype(np.float32).reshape([1,3,1,1]))
        self.std = torch.from_numpy(np.array([0.229, 0.224, 0.225]).astype(np.float32).reshape([1,3,1,1]))
        self.is_normalized = is_normalized
        
    def fix_resnet(self,fix_weight):
        self.fix_weight = fix_weight
        if fix_weight==True:
            for param in [self.resnet.parameters()]:
                for p in param:
                    p.requires_grad = False
        else:
            for param in [self.resnet.parameters()]:
                for p in param:
                    p.requires_grad = True

    def forward(self, x):
        # Prevent the bn layer from being distroied.
        if(self.fix_weight==True):
            self.layer1.eval()
            self.layer2.eval()
            self.layer3.eval()
            self.layer4.eval()
        #normalization
        if self.is_normalized:
            x -= self.mean.to(x.device)
            x /= self.std.to(x.device)
        # resnet forward
        c2 = self.layer1(x)
        c3 = self.layer2(c2)       
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return c5

if __name__ == '__main__':
    back_bone = encode(fix_weight=True).cuda().train()
    test = torch.ones([10,3,224,224]).cuda()
    ret = []
    for _ in range(3):
        ret += [back_bone(test).cpu().detach().numpy()]
    print(ret[0].all()==ret[2].all())