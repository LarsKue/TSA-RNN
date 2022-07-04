#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 09:23:39 2022

@author: janik
"""
import torch
import torch.nn as nn
import time


class LatentRNN(nn.Module):
    def __init__(self, obs_dim, latent_dim):
        super().__init__()
        self.RNN = nn.RNN(obs_dim, latent_dim, num_layers=1, nonlinearity='relu')
        self.observation_model = nn.Linear(latent_dim, obs_dim)

    def forward(self, time_series, h0):
        # The RNN class doesn't care how long the input is as long as time is the first and features the last dim
        rnn_output, h = self.RNN(time_series, h0)
        obs_output = self.observation_model(rnn_output)
        return obs_output, h


def train(model, optimizer, loss_function, epochs, batching, verbose=True):
    loss_over_time = []
    t0 = time.time()
    for i in range(epochs):
        optimizer.zero_grad()

        if batching:
            # Mini-batches of sub-sequences are drawn at random and concatenated along the batch dimension (dim 1)
            random_indices = torch.randperm(data.shape[0] - seq_length)[:seq_per_epoch]
            x = torch.cat([data[r:r + seq_length].unsqueeze(1) for r in random_indices], dim=1)
            y = torch.cat([data[r + 1:r + seq_length + 1].unsqueeze(1) for r in random_indices], dim=1)
            h0 = torch.randn((1, seq_per_epoch, hidden_size))
        else:
            # With batching=False, I still create a batching dimension, but it's of size 1
            x = data[:-1].unsqueeze(1)
            y = data[1:].unsqueeze(1)
            h0 = torch.randn((1, 1, hidden_size))

        output, _ = model(x, h0)

        epoch_loss = loss_function(output, y)
        epoch_loss.backward()
        optimizer.step()
        if i % 10 == 0 and i > 0:
            t1 = time.time()
            epochs_per_sec = 10 / (t1 - t0)
            if verbose:
                print(f"Epoch: {i} loss {epoch_loss.item()} @ {epochs_per_sec} epochs per second")
            loss_over_time.append(epoch_loss.item())
            t0 = t1

    return model, loss_over_time
