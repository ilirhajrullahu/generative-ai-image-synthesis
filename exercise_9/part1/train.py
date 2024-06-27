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


################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time
from datetime import datetime
import numpy as np

import torch
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from dataset import PalindromeDataset
from vanilla_rnn import VanillaRNN
from lstm import LSTM

# You may want to look into tensorboardX for logging
# from tensorboardX import SummaryWriter

################################################################################

def test_single_prediction_lstm(config):
    device = torch.device(config.device)
    model = LSTM(config.input_length, config.input_dim, config.num_hidden, config.num_classes, config.batch_size, device).to(device)
    dataset = PalindromeDataset(config.input_length + 1)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)
    for step, (batch_inputs, batch_targets) in enumerate(data_loader):
        print(batch_inputs.shape)
        print(batch_targets.shape)
        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)
        print('input',batch_inputs[0])
        print('truth',batch_targets[0])
        outputs = model(batch_inputs)
        _, predicted = torch.max(outputs, 1)
        print('prediction',predicted[0])
        accuracy = (predicted[0] == batch_targets[0]).sum().item()
        print('accuracy', accuracy)
        break

def test_single_prediction_rnn(config):
    device = torch.device(config.device)
    model = VanillaRNN(config.input_length, config.input_dim, config.num_hidden, config.num_classes, config.batch_size, device)
    dataset = PalindromeDataset(config.input_length + 1)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)
    for step, (batch_inputs, batch_targets) in enumerate(data_loader):
        print(batch_inputs.shape)
        print(batch_targets.shape)
        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)
        print('input',batch_inputs[0])
        print('truth',batch_targets[0])
        outputs = model(batch_inputs)
        _, predicted = torch.max(outputs, 1)
        print('prediction',predicted[0])
        accuracy = (predicted[0] == batch_targets[0]).sum().item()
        print('accuracy', accuracy)
        break

def experiment_memorization_lstm(config):
    device = torch.device(config.device)
    
    avg_accuracies = []
    
    for seq_length in range(5, 11):
        # Initialize the model

        model = LSTM(seq_length - 1, config.input_dim, config.num_hidden, config.num_classes, config.batch_size, device).to(device)
        
        # Initialize the dataset and data loader
        dataset = PalindromeDataset(seq_length)
        data_loader = DataLoader(dataset, config.batch_size, num_workers=1)
        
        # Setup the loss and optimizer
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.RMSprop(model.parameters(), lr=config.learning_rate)
        
        # Training loop
        total_accuracy = 0.0
        count = 0
        for step, (batch_inputs, batch_targets) in enumerate(data_loader):
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
                
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_targets)
                
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_norm)
            optimizer.step()

            _, predicted = torch.max(outputs, 1)
            accuracy = (predicted == batch_targets).sum().item() / config.batch_size
            total_accuracy += accuracy
            count += 1
            if step % 10 == 0:
                print(
                        "[{}] Train Step {:04d}/{:04d}, Batch Size = {}"
                        " Accuracy = {:.2f}, Loss = {:.3f}, seq_length = {}".format(
                            datetime.now().strftime("%Y-%m-%d %H:%M"),
                            step,
                            config.train_steps,
                            config.batch_size,
                            accuracy,
                            loss,
                            seq_length,
                        )
                    )

            if step == config.train_steps:
                break
        
        avg_accuracy = total_accuracy / count
        avg_accuracies.append(avg_accuracy)   
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(range(5, 11), avg_accuracies, marker='o')
    plt.xlabel('Palindrome Length')
    plt.ylabel('Accuracy')
    plt.title(f'LSTM Memorization Capability: Accuracy vs Palindrome Length with {config.train_steps} training steps')
    plt.grid(True)
    plt.savefig('lstm_memorization_accuracy.png')
    plt.show()

def experiment_memorization_rnn(config):
    device = torch.device(config.device)
    
    avg_accuracies = []
    
    for seq_length in range(5, 11):
        # Initialize the model
        model = VanillaRNN(seq_length - 1, config.input_dim, config.num_hidden, config.num_classes, config.batch_size, device).to(device)
        
        # Initialize the dataset and data loader
        dataset = PalindromeDataset(seq_length)
        data_loader = DataLoader(dataset, config.batch_size, num_workers=1)
        
        # Setup the loss and optimizer
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.RMSprop(model.parameters(), lr=config.learning_rate)
        total_accuracy = 0.0
        count = 0
        # Training loop
        for step, (batch_inputs, batch_targets) in enumerate(data_loader):
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
                
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_targets)
                
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_norm)
            optimizer.step()

            _, predicted = torch.max(outputs, 1)
            accuracy = (predicted == batch_targets).sum().item() / config.batch_size # fixme

            total_accuracy += accuracy
            count += 1
                
            if step % 10 == 0:
                print(
                        "[{}] Train Step {:04d}/{:04d}, Batch Size = {}"
                        " Accuracy = {:.2f}, Loss = {:.3f}, seq_length = {}".format(
                            datetime.now().strftime("%Y-%m-%d %H:%M"),
                            step,
                            config.train_steps,
                            config.batch_size,
                            accuracy,
                            loss,
                            seq_length,
                        )
                    )

            if step == config.train_steps:
                # If you receive a PyTorch data-loader error, check this bug report:
                # https://github.com/pytorch/pytorch/pull/9655
                break
        avg_accuracy = total_accuracy / count
        avg_accuracies.append(avg_accuracy)        

    plt.figure(figsize=(10, 6))
    plt.plot(range(5, 11), avg_accuracies, marker='o')
    plt.xlabel('Palindrome Length')
    plt.ylabel('Accuracy')
    plt.title(f'RNN Memorization Capability: Avg. Accuracy vs Palindrome Length with {config.train_steps} training steps')
    plt.grid(True)
    plt.savefig("accuracy_vs_palindrome_length.png")
    plt.show()


def train(config):

    assert config.model_type in ("RNN", "LSTM")

    # Initialize the device which to run the model on
    device = torch.device(config.device)

    print (config)
    # Initialize the model that we are going to use
    if config.model_type == "RNN":
        model = VanillaRNN(config.input_length, config.input_dim, config.num_hidden, config.num_classes, config.batch_size, device)
    elif config.model_type == "LSTM":
        model = LSTM(config.input_length, config.input_dim, config.num_hidden, config.num_classes, config.batch_size, device)

    # Initialize the dataset and data loader (note the +1)
    dataset = PalindromeDataset(config.input_length + 1)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

    # Setup the loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()  # fixme
    optimizer = torch.optim.RMSprop(model.parameters(),lr=config.learning_rate)  # fixme

    for step, (batch_inputs, batch_targets) in enumerate(data_loader):

        # Only for time measurement of step through network
        t1 = time.time()

        # Add more code here ...

        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)
        outputs = model(batch_inputs)
        loss = criterion(outputs, batch_targets)
        optimizer.zero_grad()
        loss.backward()
        ############################################################################
        # QUESTION: what happens here and why?
        ############################################################################
        torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=config.max_norm)
        ############################################################################
        optimizer.step()
        # Add more code here ...

        #loss = loss = criterion(outputs, batch_targets)  # fixme
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == batch_targets).sum().item() / config.batch_size # fixme

        # Just for time measurement
        t2 = time.time()
        examples_per_second = config.batch_size / float(t2 - t1)

        if step % 10 == 0:

            print(
                "[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
                "Accuracy = {:.2f}, Loss = {:.3f}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M"),
                    step,
                    config.train_steps,
                    config.batch_size,
                    examples_per_second,
                    accuracy,
                    loss,
                )
            )

        if step == config.train_steps:
            # If you receive a PyTorch data-loader error, check this bug report:
            # https://github.com/pytorch/pytorch/pull/9655
            break

    print("Done training.")


################################################################################
################################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument(
        "--model_type",
        type=str,
        default="RNN",
        help="Model type, should be 'RNN' or 'LSTM'",
    )
    parser.add_argument(
        "--input_length", type=int, default=10, help="Length of an input sequence"
    )
    parser.add_argument(
        "--input_dim", type=int, default=1, help="Dimensionality of input sequence"
    )
    parser.add_argument(
        "--num_classes", type=int, default=10, help="Dimensionality of output sequence"
    )
    parser.add_argument(
        "--num_hidden",
        type=int,
        default=128,
        help="Number of hidden units in the model",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Number of examples to process in a batch",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="Learning rate"
    )
    parser.add_argument(
        "--train_steps", type=int, default=10000, help="Number of training steps"
    )
    parser.add_argument("--max_norm", type=float, default=10.0)
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Training device 'cpu' or 'cuda:0'"
    )

    config = parser.parse_args()

    # Train the model
    #train(config) # run this to test exercise 1.2
    #experiment_memorization_rnn(config) # run this to test exercise 1.3
    #test_single_prediction_rnn(config)
    #experiment_memorization_lstm(config) # run this to test exercise 1.5 and 1.6
    #test_single_prediction_lstm(config)