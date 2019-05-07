# MIT License
#
# Copyright (c) 2017 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2018
# Date Created: 2018-09-04
################################################################################

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

################################################################################

def train(config):

    def acc(predictions, targets):
        hotvec = predictions.argmax(-2) == targets
        accuracy = torch.mean(hotvec.float())
        return accuracy

    # Initialize the device which to run the model on
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Initialize the dataset and data loader (note the +1)
    dataset = TextDataset(config.txt_file, config.seq_length)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=0)
    print('batch', config.batch_size)

    vocabulary_size = dataset.vocab_size
    print('vocab', vocabulary_size)
    # Initialize the model that we are going to use
    model = TextGenerationModel(config.batch_size,
                                config.seq_length,
                                vocabulary_size=vocabulary_size,
                                lstm_num_hidden=config.lstm_num_hidden,
                                lstm_num_layers=config.lstm_num_layers,
                                dropout=1-config.dropout_keep_prob,
                                device=device
                                )
    model = model.to(device)
    # Setup the loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config.learning_rate
                                 )
    print('Hi')
    acc_list = []
    loss_list = []
    step_list = []
    text_list = []
    epoch = 50
    offset = 2380
    temperature = 1
    policy = 'greedy'
    for e in range(epoch):
        #torch.save(model.state_dict(), str(e+1) + 'model.pt')
        for step, (batch_inputs, batch_targets) in enumerate(data_loader):

            # Only for time measurement of step through network
            #lr_optim.step()
            optimizer.zero_grad()
            t1 = time.time()
            inputs = torch.stack([*batch_inputs], dim=1)
            targets = torch.stack([*batch_targets], dim=1)
            inputs = inputs.to(device)
            targets = targets.to(device)
            out = model.forward(inputs)[0]
            out = out.permute(0, 2, 1)
            loss = criterion(out, targets)
            accuracy = acc(out, targets)

            torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=config.max_norm)
            loss.backward()
            optimizer.step()

            # Just for time measurement
            t2 = time.time()
            examples_per_second = config.batch_size/float(t2-t1)

            if step % config.print_every == 0:

                print('accuracy, loss, step: \n',
                        np.around(accuracy.item(), 4), np.around(loss.item(), 4),
                      step, '\n'
                )
                acc_list.append(accuracy.item())
                loss_list.append(loss.item())

                step_list.append(step + offset * e)

            if step % config.sample_every == 0:
                # Generate some sentences by sampling from the model
                generator = torch.randint(low=0, high=vocabulary_size, size=(1, 1)).to(device)
                hidden = None
                char_list = [generator.item()]
                for _ in range(config.seq_length):
                    generator, hidden = model.forward(generator, hidden)
                    if policy == 'greedy':
                        idx = torch.argmax(generator).item()
                    else:
                        temp = generator.squeeze() / temperature
                        soft = torch.softmax(temp, dim=0)
                        idx = torch.multinomial(soft, 1)[-1].item()
                    generator = torch.Tensor([idx]).unsqueeze(-1)
                    generator = generator.to(device)
                    char_list.append(idx)
                char = dataset.convert_to_string(char_list)
                with open("MyBook.txt", "a") as text_file:
                    print('Epoch. ', e, 'Stahp: ', step, '\n Output: ', char, file=text_file)

                print('Epoch. ', e, 'Stahp: ', step, '\n Output: ', char)
                text_list.append((str((step + offset * e)) + '\n' + char))

            if step == config.train_steps:
                # If you receive a PyTorch data-loader error, check this bug report:
                # https://github.com/pytorch/pytorch/pull/9655
                break

    print('Done training.')

    with open('FinalBook.txt', 'w+') as f:
        for item in text_list:
            f.write("%s\n" % item)

    # save with pandas
    header = ['accuracy', 'length', 'loss', 'step']
    savefiles = zip(acc_list, [config.seq_length]*len(acc_list), loss_list, step_list)
    df = pd.DataFrame(list(savefiles), columns=header)
    df.to_csv('GEN' + str(config.seq_length) + 'lstm.csv')

    print('I am Loaded')

    temp_list = [0.5, 1., 2.]
    policy = 'temp'
    seq_length = 30

    # Generate some sentences by sampling from the model
    for temperature in temp_list:
        generator = torch.randint(low=0, high=vocabulary_size, size=(1, 1)).to(device)
        hidden = None
        char_list = [generator.item()]
        for _ in range(seq_length):
            generator, hidden = model.forward(generator, hidden)
            if policy == 'greedy':
                idx = torch.argmax(generator).item()
            else:
                temp = generator.squeeze() / temperature
                soft = torch.softmax(temp, dim=0)
                idx = torch.multinomial(soft, 1).item()
            generator = torch.tensor([idx]).unsqueeze(-1)
            generator = generator.to(device)
            char_list.append(idx)
        char = dataset.convert_to_string(char_list)
        with open(policy + str(int(np.floor(temperature))) + "Book.txt", "w+") as text_file:
            print('Temp: ', temperature, '\n Output: ', char, file=text_file)

        print('Temp: ', temperature, '\n Output: ', char)
    print('Finito!')


################################################################################
 ################################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--txt_file', type=str, required=True, help="Path to a .txt file to train on")
    parser.add_argument('--seq_length', type=int, default=30, help='Length of an input sequence')
    parser.add_argument('--lstm_num_hidden', type=int, default=128, help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=2, help='Number of LSTM layers in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=64, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=2e-3, help='Learning rate')

    # It is not necessary to implement the following three params, but it may help training.
    parser.add_argument('--learning_rate_decay', type=float, default=0.96, help='Learning rate decay fraction')
    parser.add_argument('--learning_rate_step', type=int, default=5000, help='Learning rate step')
    parser.add_argument('--dropout_keep_prob', type=float, default=1.0, help='Dropout keep probability')

    parser.add_argument('--train_steps', type=int, default=1e6, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=5.0, help='--')

    # Misc params
    parser.add_argument('--summary_path', type=str, default="./summaries/", help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=5, help='How often to print training progress')
    parser.add_argument('--sample_every', type=int, default=100, help='How often to sample from the model')

    config = parser.parse_args()

    # Train the model
    train(config)
