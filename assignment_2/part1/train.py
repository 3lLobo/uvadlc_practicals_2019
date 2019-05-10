################################################################################
# MIT License
# 
# Copyright (c) 2018
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

import argparse
import time
from datetime import datetime
import numpy as np

<<<<<<< HEAD
from operator import itemgetter
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from dataset import PalindromeDataset
from vanilla_rnn import VanillaRNN
from lstm import LSTM

# You may want to look into tensorboardX for logging - No
import pandas as pd
=======
import torch
from torch.utils.data import DataLoader

from part1.dataset import PalindromeDataset
from part1.vanilla_rnn import VanillaRNN
from part1.lstm import LSTM

# You may want to look into tensorboardX for logging
# from tensorboardX import SummaryWriter

>>>>>>> 6482db2f27c75cbc122b668ce0719fb5aa1e3de1
################################################################################

def train(config):

<<<<<<< HEAD
    def acc(predictions, targets):
        hotvec = predictions.argmax(-1) == targets
        accuracy = torch.mean(hotvec.float())
        return accuracy

    assert config.model_type in ('RNN', 'LSTM')

    # Initialize the device which to run the model on

    model_type = config.model_type
    input_length = config.input_length
    input_dim = config.input_dim
    num_classes = config.num_classes
    num_hidden = config.num_hidden
    batch_size = config.batch_size
    lr = config.learning_rate
    train_steps = config.train_steps
    max_norm = config.max_norm

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    acc_list = []
    loss_list = []
    epoch_list = []

    # Run experiment 5 times for significant results
    for _ in range(3):
        # Initialize the model that we are going to use
        if model_type == 'RNN':
            model = VanillaRNN(input_length, input_dim, num_hidden, num_classes, batch_size, device=device)
            model.to(device)
        elif model_type =='LSTM':
            model = LSTM(input_length, input_dim, num_hidden, num_classes, batch_size, device=device)
            model.to(device)

        # Initialize the dataset and data loader (note the +1)
        dataset = PalindromeDataset(input_length+1)
        data_loader = DataLoader(dataset, batch_size, num_workers=0)

        # Setup the loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, alpha=0.99,
                                        eps=1e-08, weight_decay=0, momentum=0,
                                        centered=False
                                        )
        print('start training')

        for step, tpl in enumerate(data_loader):
            (batch_inputs, batch_targets) = tpl
            # Only for time measurement of step through network
            t1 = time.time()
            # Add more code here ...

            #tensor_input = torch.Tensor(batch_inputs, dtype=torch.float, device=device)
            #tensor_targets = torch.Tensor(batch_targets, dtype=torch.long, device=device)
            batch_targets = batch_targets.to(device)
            output = model.forward(batch_inputs)
            loss = criterion(output, batch_targets)
            accuracy = acc(output, batch_targets)

            optimizer.zero_grad()
            loss.backward()

            ############################################################################
            # QUESTION: what happens here and why?
            # ANSWER:   It scales the gradient. With each layer backtracked the gradiend gets amplified.
            #           This can result in an exploding gradient.
            #           To avoid this, the gradient is clipped to the max_norm
            ############################################################################
            torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=max_norm)
            ############################################################################

            # Add more code here ...

            optimizer.step()

            # Just for time measurement
            t2 = time.time()
            #examples_per_second = config.batch_size/float(t2-t1)

            if step % 10 == 0:
                epoch_list.append(step)
                acc_list.append(accuracy.item())
                loss_list.append(loss.item())
                print('acc', acc_list[-1])
                #print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
                #      "Accuracy = {:.2f}, Loss = {:.3f}".format(
                #        datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                #        train_steps, batch_size, examples_per_second,
                #        accuracy, loss
                #))

            if step == train_steps:

                print('Mean', np.mean(acc_list))
                print('std', np.std(acc_list))
                print('Run done!')
                # If you receive a PyTorch data-loader error, check this bug report:
                # https://github.com/pytorch/pytorch/pull/9655
                break


    # save with pandas too
    header = ['accuracy', 'length', 'loss', 'epoch']
    savefiles = zip(acc_list, [input_length+1]*len(acc_list), loss_list, epoch_list)
    df = pd.DataFrame(list(savefiles), columns=header)
    df.to_csv(model_type + str(input_length) + 'results.csv')
=======
    assert config.model_type in ('RNN', 'LSTM')

    # Initialize the device which to run the model on
    device = torch.device(config.device)

    # Initialize the model that we are going to use
    model = None  # fixme

    # Initialize the dataset and data loader (note the +1)
    dataset = PalindromeDataset(config.input_length+1)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

    # Setup the loss and optimizer
    criterion = None  # fixme
    optimizer = None  # fixme

    for step, (batch_inputs, batch_targets) in enumerate(data_loader):

        # Only for time measurement of step through network
        t1 = time.time()

        # Add more code here ...

        ############################################################################
        # QUESTION: what happens here and why?
        ############################################################################
        torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=config.max_norm)
        ############################################################################

        # Add more code here ...

        loss = np.inf   # fixme
        accuracy = 0.0  # fixme

        # Just for time measurement
        t2 = time.time()
        examples_per_second = config.batch_size/float(t2-t1)

        if step % 10 == 0:

            print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
                  "Accuracy = {:.2f}, Loss = {:.3f}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                    config.train_steps, config.batch_size, examples_per_second,
                    accuracy, loss
            ))

        if step == config.train_steps:
            # If you receive a PyTorch data-loader error, check this bug report:
            # https://github.com/pytorch/pytorch/pull/9655
            break

>>>>>>> 6482db2f27c75cbc122b668ce0719fb5aa1e3de1
    print('Done training.')


 ################################################################################
 ################################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--model_type', type=str, default="RNN", help="Model type, should be 'RNN' or 'LSTM'")
    parser.add_argument('--input_length', type=int, default=10, help='Length of an input sequence')
    parser.add_argument('--input_dim', type=int, default=1, help='Dimensionality of input sequence')
    parser.add_argument('--num_classes', type=int, default=10, help='Dimensionality of output sequence')
    parser.add_argument('--num_hidden', type=int, default=128, help='Number of hidden units in the model')
    parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--train_steps', type=int, default=10000, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=10.0)
    parser.add_argument('--device', type=str, default="cuda:0", help="Training device 'cpu' or 'cuda:0'")

    config = parser.parse_args()

    # Train the model
    train(config)