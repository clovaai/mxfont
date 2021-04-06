"""
MX-Font
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

from pathlib import Path
from itertools import chain
import random

import torch
from torch.utils.data import Dataset

from .ttf_utils import read_font, render


class TTFTrainDataset(Dataset):
    def __init__(self, data_dir, primals, decomposition, transform=None,
                 n_in_s=3, n_in_c=3, source_font=None):

        self.data_dir = data_dir
        self.primals = primals
        self.decomposition = decomposition

        self.key_font_dict, self.key_char_dict = load_data_list(data_dir, char_filter=list(self.decomposition))
        self.char_key_dict = {}
        for key, charlist in self.key_char_dict.items():
            for char in charlist:
                self.char_key_dict.setdefault(char, []).append(key)

        self.key_char_dict, self.char_key_dict = self.filter_chars()
        self.data_list = [(key, char) for key, chars in self.key_char_dict.items() for char in chars]

        self.keys = sorted(self.key_font_dict)
        self.chars = sorted(set.union(*map(set, self.key_char_dict.values())))

        self.transform = transform

        self.n_in_s = n_in_s
        self.n_in_c = n_in_c
        self.n_chars = len(self.chars)
        self.n_fonts = len(self.keys)

    def filter_chars(self):
        char_key_dict = {}
        for char, keys in self.char_key_dict.items():
            num_keys = len(keys)
            if num_keys > 1:
                char_key_dict[char] = keys
            else:
                pass

        filtered_chars = list(char_key_dict)
        key_char_dict = {}
        for key, chars in self.key_char_dict.items():
            key_char_dict[key] = list(set(chars).intersection(filtered_chars))

        return key_char_dict, char_key_dict

    def __getitem__(self, index):
        key, char = self.data_list[index]
        font = self.key_font_dict[key]
        fidx = self.keys.index(key)
        cidx = self.chars.index(char)

        trg_img = self.transform(render(font, char))
        trg_dec = [self.primals.index(x) for x in self.decomposition[char]]

        style_chars = sample([c for c in self.key_char_dict[key] if c != char], self.n_in_s)
        style_imgs = torch.stack([self.transform(render(font, c)) for c in style_chars])
        style_decs = [[self.primals.index(x) for x in self.decomposition[c]] for c in style_chars]

        char_keys = sample([k for k in self.char_key_dict[char] if k != key], self.n_in_c)

        char_imgs = torch.stack([self.transform(render(self.key_font_dict[k], char)) for k in char_keys])
        char_decs = [trg_dec] * self.n_in_c
        char_fids = [self.keys.index(_k) for _k in char_keys]

        ret = {
            "trg_imgs": trg_img,
            "trg_decs": trg_dec,
            "trg_fids": torch.LongTensor([fidx]),
            "trg_cids": torch.LongTensor([cidx]),
            "style_imgs": style_imgs,
            "style_decs": style_decs,
            "style_fids": torch.LongTensor([fidx]*self.n_in_s),
            "char_imgs": char_imgs,
            "char_decs": char_decs,
            "char_fids": torch.LongTensor(char_fids)
        }

        return ret

    def __len__(self):
        return len(self.data_list)

    @staticmethod
    def collate_fn(batch):
        _ret = {}
        for dp in batch:
            for key, value in dp.items():
                saved = _ret.get(key, [])
                _ret.update({key: saved + [value]})

        ret = {
            "trg_imgs": torch.stack(_ret["trg_imgs"]),
            "trg_decs": _ret["trg_decs"],
            "trg_fids": torch.cat(_ret["trg_fids"]),
            "trg_cids": torch.cat(_ret["trg_cids"]),
            "style_imgs": torch.stack(_ret["style_imgs"]),
            "style_decs": [*chain(*_ret["style_decs"])],
            "style_fids": torch.stack(_ret["style_fids"]),
            "char_imgs": torch.stack(_ret["char_imgs"]),
            "char_decs": [*chain(*_ret["char_decs"])],
            "char_fids": torch.stack(_ret["char_fids"])
        }

        return ret


class TTFValDataset(Dataset):
    def __init__(self, data_dir, source_font, char_filter, n_ref=4, n_gen=20, transform=None):

        self.data_dir = data_dir
        self.source_font = read_font(source_font) if source_font is not None else None
        self.n_ref = n_ref
        self.n_gen = n_gen

        self.key_font_dict, self.key_char_dict = load_data_list(data_dir, char_filter=char_filter)
        if self.source_font is None:
            self.char_key_dict = {}
            for key, charlist in self.key_char_dict.items():
                for char in charlist:
                    self.char_key_dict.setdefault(char, []).append(key)

            self.key_char_dict, self.char_key_dict = self.filter_chars()
        self.ref_chars, self.gen_chars = self.sample_ref_gen_chars(self.key_char_dict)

        self.gen_char_dict = {k: self.gen_chars for k in self.key_font_dict}
        self.data_list = [(key, char) for key, chars in self.gen_char_dict.items() for char in chars]
        self.transform = transform

    def sample_ref_gen_chars(self, key_char_dict):
        common_chars = sorted(set.intersection(*map(set, key_char_dict.values())))
        sampled_chars = sample(common_chars, self.n_ref+self.n_gen)
        ref_chars = sampled_chars[:self.n_ref]
        gen_chars = sampled_chars[self.n_ref:]

        return ref_chars, gen_chars

    def __getitem__(self, index):
        key, char = self.data_list[index]
        font = self.key_font_dict[key]

        ref_imgs = torch.stack([self.transform(render(font, c))
                                for c in self.ref_chars])

        if self.source_font is not None:
            source_font = self.source_font
        else:
            source_key = random.choice(self.char_key_dict[char])
            source_font = self.key_font_dict[source_key]

        source_img = self.transform(render(source_font, char))
        trg_img = self.transform(render(font, char))

        ret = {
            "style_imgs": ref_imgs,
            "source_imgs": source_img,
            "fonts": key,
            "chars": char,
            "trg_imgs": trg_img
        }

        return ret

    def __len__(self):
        return len(self.data_list)

    @staticmethod
    def collate_fn(batch):
        _ret = {}
        for dp in batch:
            for key, value in dp.items():
                saved = _ret.get(key, [])
                _ret.update({key: saved + [value]})

        ret = {
            "style_imgs": torch.stack(_ret["style_imgs"]),
            "source_imgs": torch.stack(_ret["source_imgs"]),
            "fonts": _ret["fonts"],
            "chars": _ret["chars"],
            "trg_imgs": torch.stack(_ret["trg_imgs"])
        }

        return ret


def sample(population, k):
    if len(population) < k:
        sampler = random.choices
    else:
        sampler = random.sample
    sampled = sampler(population, k=k)
    return sampled


def load_data_list(data_dir, char_filter=None):
    font_paths = sorted(Path(data_dir).glob("*.ttf"))

    key_font_dict = {}
    key_char_dict = {}

    for font_path in font_paths:
        font = read_font(font_path)
        key_font_dict[font_path.stem] = font

        with open(str(font_path).replace(".ttf", ".txt")) as f:
            chars = f.read()

        if char_filter is not None:
            chars = set(chars).intersection(char_filter)
        key_char_dict[font_path.stem] = list(chars)

    return key_font_dict, key_char_dict
