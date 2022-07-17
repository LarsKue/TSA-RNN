#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 10:15:54 2022

@author: janik
"""

import names_dataset as nd
import name_generator as ng

dataset = nd.NamesDataset(allow_unicode=False)

''' TODO: change the model hyperparameters if you like '''
# ~1M parameters already works really well
hidden_size = 256
layers = 2
dropout = 0.25

model = ng.NameGenerator(dataset, hidden_size, layers, dropout)

''' TODO: make the training parameters such that training is fast enough '''
# lower batch size is usually better (more weight updates)
# this trains at roughly 2.7 epochs per second on an RTX2080
# and achieves a test loss of ~0.60, which, after experimentation, seems optimal
epochs = 120
batch_size = 32
batches_per_epoch = 128
lr_milestones = [80, 100]
lr_gamma = 0.1

''' TODO: generate some names '''
model, losses = ng.train(model, dataset, epochs, lr_milestones, lr_gamma, batch_size, batches_per_epoch)
# model.generate('ger', 10)

model.eval()
with open("names.txt", "w+") as f:
    for i, language in enumerate(dataset.language_set):
        print(f"Language {i}: {language}")
        f.write(f"Language: {language.capitalize()}:\n")
        f.writelines([
            f"    {model.generate(language, 10).strip()}\n" for _ in range(10)
        ])
        f.write("\n")
