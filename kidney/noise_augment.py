import torch


class AddGaussianNoise:
    def __init__(self, mean=0., std=1., clip=True):
        self.std = std
        self.mean = mean
        self.clip = clip
        
    def __call__(self, tensor):
        if self.clip:
            return torch.clamp(tensor + torch.randn(tensor.size()) * self.std + self.mean, 0, 1)
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)