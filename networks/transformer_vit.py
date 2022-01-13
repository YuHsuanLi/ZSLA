import torch
from torch import nn, einsum
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import random
#from networks import utils

SEED = 2021
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
"""
reference: https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py 
"""
'''
def tensor_to_gpu(t, device=None):
    t = t.to(device)
    return t
'''
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        
        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class logic_gate(nn.Module):
    def __init__(self, Ch, n_layers, device=None, pool = 'first', vit_gate_allow_multi_input = False, is_abs=True):
        '''
        Ch: int, the feature length
        n_layers: int, the number of the layer of the logic gates, Number of or/and/not hidden layers
        device: str, the device name
        '''
        super().__init__()
        self.Ch = Ch
        self.n_layers =  n_layers
        self.device = device
        self.and_layer =  Transformer(dim=Ch, depth=n_layers, heads=16, dim_head=64, mlp_dim=Ch, dropout=0.1).to(self.device)  
        self.or_layer =  Transformer(dim=Ch, depth=n_layers, heads=16, dim_head=64, mlp_dim=Ch, dropout=0.1).to(self.device)
        self.not_layer =  Transformer(dim=Ch, depth=n_layers, heads=16, dim_head=64, mlp_dim=Ch, dropout=0.1).to(self.device)
        cls_token_v = nn.Parameter(torch.randn(1, 1, Ch))
        self.register_parameter('cls_token' , cls_token_v)
        #print(cls_token_v)
        #exit(0)
        self.is_abs = is_abs
    def uniform_size_multi_input(self, vectors, train):
        '''
        vector1: tensor[?XCh], one of the input of 'and' & 'or' gate 
        vector2: tensor[?XCh], the other input of 'and' & 'or' gate
        train: bool, is in training stage or not
        return tensor[?XCh], tensor[?XCh]
        '''
        
        len_of_vector_size = np.array([len(vector.size()) for vector in vectors])
        biggest_vector = vectors[np.argmax(len_of_vector_size)] 
        vectors = [vector.expand_as(biggest_vector) for vector in vectors]
        if train:
            random.shuffle(vectors)
        return vectors
    def uniform_size(self, vector1, vector2, train):
        '''
        vector1: tensor[?XCh], one of the input of 'and' & 'or' gate 
        vector2: tensor[?XCh], the other input of 'and' & 'or' gate
        train: bool, is in training stage or not
        return: tensor[?XCh], tensor[?XCh]
        '''
        if len(vector1.size()) < len(vector2.size()):
            vector1 = vector1.expand_as(vector2)
        else:
            vector2 = vector2.expand_as(vector1)
        if train:
            r12 = torch.Tensor(vector1.size()[:-1]).uniform_(0, 1).bernoulli()
            r12 = r12.unsqueeze(-1).to(vector1.device)
            #r12 = tensor_to_gpu(r12, self.device).unsqueeze(-1).to(vector1.device)
            new_v1 = r12 * vector1 + (-r12 + 1) * vector2
            new_v2 = r12 * vector2 + (-r12 + 1) * vector1
            return new_v1, new_v2
        return vector1, vector2
    
    def logic_and(self, vector1_, vector2_, train=True):
        '''
        vector1: tensor[?XCh], one of the input of 'and' gate 
        vector2: tensor[?XCh], the other input of 'and' gate
        train: bool, is in training stage or not
        return: tensor[?XCh]
        '''        
        #if self.vit_gate_allow_multi_input == False:
        vector1, vector2 = self.uniform_size(vector1_, vector2_, train)
        do_squeeze_before_return = False
        if len(vector1.shape)<2:
            do_squeeze_before_return = True
            vector1 = vector1.unsqueeze(0)
            vector2 = vector2.unsqueeze(0)        

        vector = torch.cat((vector1.unsqueeze(1), vector2.unsqueeze(1)), dim=1)
        # reshape cls_token so that it fit the batch size and concat it to the input(vector)
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = vector.shape[0]).to(self.device)
        vector = torch.cat((cls_tokens, vector), dim=1)

        vector = vector.to(self.device)
        vector = self.and_layer(vector)
        vector = vector[:, 0]
        if do_squeeze_before_return:
            vector = vector.squeeze()
        if self.is_abs:
            return torch.abs(vector)
        return vector
