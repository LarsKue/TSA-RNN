#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 22:33:36 2022

@author: janik
"""

import torch as tc
from torch.utils.data import Dataset
import os
import re


class NamesDataset(Dataset):

    def __init__(self, allow_unicode=False):
        super().__init__()
        special_chars = {'ß': 'ss', 'àáã': 'a', 'ä': 'ae', 'ç': 'c', 'èéê': 'e',
                         'ìí': 'i', 'ñ': 'n', 'òóõ': 'o', 'ö': 'oe', 'ùú': 'u',
                         'ü': 'ue', 'ą': 'a', 'ł': 'l', 'ń': 'n', 'ś': 's', 'ż': 'z'}
        self.names = []
        self.languages = []
        complete_string = ''
        for file in os.listdir('names'):
            path = os.path.join('names', file)
            with open(path, 'r') as f:
                text = f.read()
            text = re.sub('[,-/1:]', '', text)
            text = text.lower()
            if not allow_unicode:
                for key, item in special_chars.items():
                    text = re.sub(f'[{key}]', item, text)
            complete_string += text
            names = text.split('\n')
            names = [n for n in names if len(n) > 0]
            self.names.extend(names)
            self.languages.extend([file[:-4].lower()] * len(names))

        self.alphabet = sorted(set(complete_string.replace('\n', '')))
        self.eos_index = len(self.alphabet)
        self.language_set = sorted(set(self.languages))
        self.complete_string = complete_string
        self.max_len = max([len(s) for s in self.names]) + 1

        self.targets = [self.encode_target(s) for s in self.names]
        self.inputs = [self.encode_input(s) for s in self.names]
        self.langs = [self.encode_language(l) for l in self.languages]

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        return self.inputs[i], self.targets[i], self.langs[i]

    def encode_target(self, string):
        encoded = tc.tensor([self.alphabet.index(s) for s in string[1:]])
        encoded = tc.cat([encoded, self.eos_index * tc.ones(self.max_len - len(string))])
        return encoded.long()

    def encode_input(self, string, pad=True):
        eye = tc.eye(len(self.alphabet) + 1)
        encoded = tc.cat([eye[self.alphabet.index(s)].unsqueeze(0) for s in string], dim=0)
        if pad:
            encoded = tc.cat([encoded, eye[-1].unsqueeze(0).repeat((self.max_len - len(string) - 1, 1))], dim=0)
        return encoded

    def decode_output(self, tensor_of_indices):
        chars = [self.alphabet[i] for i in tensor_of_indices if i != self.eos_index]
        return ''.join(chars)

    def encode_language(self, language_name):
        eye = tc.eye(len(self.language_set))
        encoded = eye[self.language_set.index(language_name)]
        return encoded

    def autofill_language(self, inp):
        inp = inp.lower()
        matches = [l for l in self.language_set if l.startswith(inp)]
        if len(matches) > 1:
            raise RuntimeError(f'Multiple possible language matches {matches}')
        elif len(matches) == 0:
            raise RuntimeError('No langugae match')
        else:
            return matches[0]


if __name__ == '__main__':
    ds = NamesDataset()
