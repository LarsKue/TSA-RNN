#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 09:47:04 2022

@author: janik
"""

import torch as tc
import torch.nn as nn
import torch.nn.functional
import torch.optim as optim
import torch.utils.data as tcdata
import matplotlib.pyplot as plt
import time


device = "cuda"


class NameGenerator(nn.Module):
    def __init__(self, dataset, hidden_size, layers=1, dropout=0):
        super().__init__()
        self.dataset = dataset
        self.letter_size = len(dataset.alphabet) + 1
        self.lang_size = len(dataset.language_set)
        input_size = self.letter_size + self.lang_size
        self.latent_model = nn.LSTM(input_size, hidden_size, layers, dropout=dropout, batch_first=True).to(device)
        self.output_layer = nn.Linear(hidden_size, self.letter_size).to(device)

        self.probability_layer = nn.LogSoftmax(dim=-1).to(device)

        n_params = sum([p.data.numel() for p in self.parameters() if p.requires_grad])
        print(f"Creating model with {n_params:.2e} parameters.")

    def forward(self, x, s):
        x = x.to(device)
        s = s.to(device)

        s = s.unsqueeze(1).repeat((1, x.shape[1], 1))
        combined_input = tc.cat((x, s), dim=-1)
        # Input dimensions: (batch * time * features)
        out, _ = self.latent_model(combined_input)
        out = self.output_layer(out)
        out = self.probability_layer(out)
        return out.cpu()

    def generate(self, language, max_length: int):
        first_letter = self.dataset.alphabet[tc.randint(2, len(self.dataset.alphabet), size=(1,))]
        x = self.dataset.encode_input(first_letter, pad=False)
        s = self.dataset.encode_language(self.dataset.autofill_language(language))
        s = s.unsqueeze(0).repeat((x.shape[0], 1))
        combined_input = tc.cat((x, s), dim=-1).to(device)
        out = tc.zeros(max_length, dtype=tc.int64)
        # This Identity matrix will be used to construct one-hot vectors from it
        eye = tc.eye(len(self.dataset.alphabet) + 1)

        h, c = None, None
        for k in range(max_length):
            if k == 0:
                y, (h, c) = self.latent_model(combined_input)
                continue
            else:
                y, (h, c) = self.latent_model(combined_input, (h, c))
            y = self.output_layer(y[-1])
            y = nn.functional.softmax(y)
            choice = tc.multinomial(y, 1)
            if choice == self.letter_size - 1:
                out = out[:k]
                break
            out[k] = choice
            x = eye[out[k]].unsqueeze(0)
            combined_input = tc.cat((x, s), dim=-1).to(device)

        out = self.dataset.decode_output(out)
        return out


def train(model, dataset, epochs, lr_milestones, lr_gamma, batch_size, batches_per_epoch, print_step=10):
    test_len = len(dataset) // 10
    train_ds, test_ds = tcdata.random_split(dataset, (len(dataset) - test_len, test_len))
    train_dl = tcdata.DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_dl = tcdata.DataLoader(test_ds, batch_size=batch_size, shuffle=False, pin_memory=True)

    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters())
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer=optimizer,
        milestones=lr_milestones,
        gamma=lr_gamma,
    )
    losses = []
    t0 = time.time()

    for ep in range(epochs):
        epoch_loss = 0
        # This is the train loop over several batches per epoch
        i = None

        """ WHY wasn't this and model.eval() already in the code when you provide a dropout option """
        model.train()
        for i, (input_seq, target_seq, language) in enumerate(train_dl):
            optimizer.zero_grad()
            output = model(input_seq, language)

            # Q: why does this have the wrong shape by default?
            # could easily just transpose the output from model.forward
            output = output.movedim(1, -1)

            # print(output.shape, target_seq.shape)
            # assert output.shape == (batch_size, 29, 19)
            # assert target_seq.shape == (batch_size, 19)

            batch_loss = loss_function(output, target_seq)
            batch_loss.backward()
            optimizer.step()

            epoch_loss += batch_loss
            if i + 1 == batches_per_epoch:
                break
        epoch_loss /= (i + 1)
        scheduler.step()

        # This is the test loop, we use tc.no_grad so that the computations are not recorded
        model.eval()
        with tc.no_grad():
            test_loss = 0
            for j, (input_seq, target_seq, language) in enumerate(test_dl):
                output = model(input_seq, language)
                batch_loss = loss_function(output.permute((0, 2, 1)), target_seq)
                test_loss += batch_loss
                if j + 1 == batches_per_epoch:
                    break
            test_loss /= (j + 1)

            losses.append(tc.tensor((epoch_loss, test_loss)).unsqueeze(1))

        if print_step > 0 and ep % print_step == 0 and ep > 0:
            t1 = time.time()
            eps = print_step / (t1 - t0)
            print(f'Epoch {ep} @ {eps:.2} epochs per second. Train loss: {epoch_loss:.7}, test loss: {test_loss:.7}')
            t0 = t1

    losses = tc.cat(losses, dim=1).T
    plt.figure()
    plt.plot(losses)
    plt.legend(('train loss', 'test_loss'))

    return model, losses
