
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
from datetime import datetime
import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd

from dataset import TextDataset
from model import TextGenerationModel

# Script for experiments with Temperature on a pre-trained LSTM model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

WonderLand = torch.load('model/49tunedmodelCP.pt', map_location=torch.device(device))

print('I am Loaded')

temp_list = [0., 0.5, 1., 2.]
policy = 'temp'
seq_length = 30
txt_file = 'alice.txt'

dataset = TextDataset(txt_file, seq_length)

test_idx = dataset.convert_to_idx('A')
print(test_idx)
test_idx = dataset.convert_to_idx('Alice')
print(test_idx)

vocabulary_size = dataset.vocab_size

# Generate some sentences by sampling from the model
for temperature in temp_list:
    generator = torch.randint(low=0, high=vocabulary_size, size=(1, 1)).to(device)
    hidden = None
    char_list = [generator.item()]
    for _ in range(seq_length):
        generator, hidden = WonderLand.forward(generator, hidden)
        if policy == 'greedy':
            idx = torch.argmax(generator).item()
        else:
            temp = temperature * generator.squeeze()
            soft = torch.softmax(temp, dim=0)
            idx = torch.multinomial(soft, 1)[-1].item()
        generator = torch.tensor([idx]).unsqueeze(-1)
        generator = generator.to(device)
        char_list.append(idx)
    char = dataset.convert_to_string(char_list)
    with open("Temp" + str(int(np.floor(temperature))) + "Book.txt", "w+") as text_file:
        print('Temp: ', temperature, '\n Output: ', char, file=text_file)

    print('Temp: ', temperature, '\n Output: ', char)

