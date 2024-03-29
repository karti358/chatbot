from torch.utils.data import DataLoader, default_collate
from .dataset import RedditDataset

class RedditDataLoader(DataLoader):
    """
    Reddit Comments Dataloader
    """
    def __init__(self,
                 tokenizer = None,
                 filepaths = [],
                 batch_size = 16,
                 sequence_length = 50,
                 num_workers=1,
                 purpose = "training",
                 collate_fn = default_collate):
        
        self.filepaths = filepaths
        self.batch_size=batch_size
        self.purpose = purpose
        self.collate_fn = collate_fn
        self.num_workers = num_workers
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length

        self.dataset = RedditDataset(tokenizer = self.tokenizer, filepaths = self.filepaths, purpose = self.purpose, sequence_length = self.sequence_length)

        self.init_kwargs = {
            "dataset":self.dataset,
            "batch_size":self.batch_size,
            "collate_fn":self.collate_fn,
            "num_workers":self.num_workers,
            "pin_memory":True
        }
        super().__init__(**self.init_kwargs)


