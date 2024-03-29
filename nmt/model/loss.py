import torch
import torch.nn.functional as F

def categorical_crossentropy(inputs, target):
    inputs = F.softmax(inputs, dim = -1)
    
    scores = -torch.sum(inputs * torch.log(target))
    return scores / torch.prod(torch.tensor(inputs.shape))
