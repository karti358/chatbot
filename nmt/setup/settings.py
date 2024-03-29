import os
from pathlib import Path
from torch.utils.data import default_collate

trainer_dataloader_dict = {
        "batch_size" : 16,
        "num_workers": 1,
        "purpose" : "training",
        "collate_fn" : default_collate
    }

encoder_dict = {
        "vocab_size" : 15000,
        "latent_dim" : 1000,
        "num_enc_layers": 1,
        "dropout": 0,
        "bidirectional": True
    }

decoder_dict = {
        "vocab_size" : 15000,
        "latent_dim" : 1000,
        "num_enc_layers": 1,
        "dropout":  0,
        "bidirectional" : True,
    }

attn_dict = {
        "concat_dim" : 4 * 1000 if encoder_dict["bidirectional"] else 2 * 1000,
        "latent_dim" : 512
    }

inference_dataloader_dict =  {
        "batch_size" : 16,
        "num_workers": 1,
        "purpose" : "training",
        "collate_fn" : default_collate
    }

######## DO NOT TOUCH ANYTHING BELOW ########

# self.datalaoder = RedditDataLoader(
#             filenames = dataloader_dict.get("filenames", None),
#             batch_size = dataloader_dict.get("batch_size", 16),
#             num_workers = dataloader_dict.get("num_workers", 1),
#             purpose = dataloader_dict.get("purpose", "training"),
#             collate_fn = dataloader_dict.get("collate_fn", default_collate)
#             )

#         self.encoder = Encoder(
#             vocab_size = encoder_dict.get("vocab_size", 15000),
#             latent_dim = encoder_dict.get("latent_dim", 1000),
#             num_enc_layers = encoder_dict.get("num_enc_layers", 1),
#             dropout = encoder_dict.get("dropout", 0),
#             bidirectional = encoder_dict.get("bidirectional", True),

#             padding_idx = training_dict.get("padding_idx", 15003)
#             )

#         self.attn_block = BahdanauAttention(
#             concat_dim = attn_dict.get("concat_dim", 4 * encoder_dict.get("latent_dim", 1000) if encoder_dict.get("bidirectional", True) else 2 * encoder_dict.get("latent_dim", 1000)),
#             latent_dim = attn_dict.get("latent_dim", 512)
#         )


#         self.decoder = Decoder(
            
#             )