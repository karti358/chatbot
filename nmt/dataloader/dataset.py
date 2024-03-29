import numpy as np
from torch.utils.data import IterableDataset
from pathlib import Path
import torch
import torch.nn.functional as F

class RedditDataset(IterableDataset):
    def __init__(self,
                 tokenizer = None,
                 filepaths = [],
                 purpose = "training",
                 sequence_length = 50,
                 **kwargs):
        """
        filenames(list): List of filenames to fetch data from
        testing(bool): Whether to fetch the data for testing or training 
        """
        super().__init__()

        self.purpose = purpose
        self.tokenizer = tokenizer
        self.filepaths = filepaths
        self.sequence_length = sequence_length
        self.padding_idx = self.tokenizer.special_tokens.get("<|padding|>")
        
    def setlength(self, ids):
        if ids.shape[0] >= self.sequence_length:
            return ids[:self.sequence_length]
            
        return torch.cat( [ ids, torch.full ( (self.sequence_length - len(ids), ),  self.padding_idx, dtype = torch.int64) ], dim = 0)

    def get_ids(self, x):
        return self.setlength( torch.tensor(self.tokenizer.encode(x, allowed_special = "all") , dtype = torch.int64) )

    def get_data(self, par, comm):
        ret_par = "<|start|> " + par.strip() + " <|end|>"

        if self.purpose != "testing":
            inp_comm = comm.strip() + " <|end|>"
            out_comm = "<|start|> " + comm.strip()

            return self.get_ids(ret_par), self.get_ids(inp_comm), self.get_ids(out_comm)
        else:
            inp_comm = "<|start|> " + comm.strip()
            return self.get_ids(ret_par), self.get_ids(inp_comm)

    def __iter__(self):
        for filepath in self.filepaths:
            with open(filepath, "r", encoding="utf-8") as f:
                for text in f:
                    par, comm = text.split("<||>")

                    yield self.get_data(par, comm)
