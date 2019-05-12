import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
#from a3_gan_template import Encoder
#from a3_gan_template import Decoder
import a3_gan_template

from datasets.bmnist import bmnist
import numpy as np
from scipy.stats import norm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.load('modelsave.pt')
model.eval()
model_im = model.sample(9)[0]
plt.imshow(model_im[0, :, :])
# TODO: Fix row size
im_grid = make_grid(model_im, nrow=3)
print('Grid:', im_grid.shape)
plt.imshow(im_grid.permute(1, 2, 0))
plt.axis('off')
plt.savefig('VAEsample' + str(epoch) + '.png')