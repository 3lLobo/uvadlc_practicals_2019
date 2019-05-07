
import torch
import torch.nn as nn
import numpy as np

from dataset import TextDataset
from model import TextGenerationModel

device = 'cpu'


dataset = TextDataset('alice.txt', 30)
vocabulary_size = dataset.vocab_size

model = TextGenerationModel(64, 30, vocabulary_size=vocabulary_size,
                                lstm_num_hidden=128,
                                lstm_num_layers=2,
                                dropout=0,
                                device=device
                                )

model.load_state_dict(torch.load('50model.pt', map_location=device))
model.eval()

print('model sayz:')

print('I am Loaded')

temp_list = [0.5, 1., 2.]
policy_list = ['greedy', 'temp']
seq_length = 111
alice_string = list('Alice')

# Generate some sentences by sampling from the model
for policy in policy_list:
    for temperature in temp_list:
        char_list = []
        hidden = None
        for alice in alice_string:
            idx = dataset.convert_to_idx(alice)
            char_list.append(idx)
            generator = torch.tensor([idx]).unsqueeze(-1)
            generator = generator.to(device)
            generator, hidden = model.forward(generator, hidden)

        for _ in range(seq_length):
            if policy == 'greedy':
                idx = torch.argmax(generator).item()
            else:
                temp = generator.squeeze() / temperature
                soft = torch.softmax(temp, dim=0)
                idx = torch.multinomial(soft, 1).item()
            generator = torch.tensor([idx]).unsqueeze(-1)
            generator = generator.to(device)
            generator, hidden = model.forward(generator, hidden)
            char_list.append(idx)
        char = dataset.convert_to_string(char_list)
        with open("BonusTemp" + str(int(np.floor(temperature))) + "Book.txt", "w+") as text_file:
            print(policy + ': ', temperature, '\n Output: ', char, file=text_file)

        print(policy + ': ', temperature, '\n Output: ', char)
print('Finito!')