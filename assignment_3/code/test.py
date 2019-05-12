import torch

a = torch.tensor([0.5, 0.5, 0.991])
b = torch.tensor([1., 0., 1.])

dis = torch.nn.functional.binary_cross_entropy(a, b)
print(dis.shape)


