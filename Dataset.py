import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, lang_src, lang_tgt, seq_len):
        super(BilingualDataset, self).__init__()
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.lang_src = lang_src
        self.lang_tgt = lang_tgt
        self.seq_len = seq_len

        self.sos_token = torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_src.token_to_id('[PAD]')], dtype=torch.int64)



    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        src_target_pair = self.ds[idx]
        src_text = src_target_pair['translation'][self.lang_src]
        tgt_text = src_target_pair['translation'][self.lang_tgt]

        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        enc_num_pads = self.seq_len - len(enc_input_tokens) - 2
        dec_num_pads = self.seq_len - len(dec_input_tokens) - 1

        if enc_num_pads < 0 or dec_num_pads < 0:
            raise ValueError("Sequence Length is too short")

        # Add SOS and EOS tokens
        enc_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_pads, dtype=torch.int64)
            ]
        )

        # Add SOS token
        dec_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_pads, dtype=torch.int64)
            ]
        )

        # Add EOS token to the label
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_pads, dtype=torch.int64)
            ]
        )

        # Ensure that the sequence length is correct
        assert enc_input.shape[0] == self.seq_len
        assert dec_input.shape[0] == self.seq_len
        assert label.shape[0] == self.seq_len

        return {
            "encoder_input": enc_input, # (Seq_len)
            "decoder_input": dec_input, # (Seq_len)
            "encoder_mask": (enc_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, Seq_len)
            "decoder_mask": (dec_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(dec_input.size(0)),  # (1, Seq_len) & (1, Seq_len, Seq_len)
            "label": label, # (Seq_len)
            "src_text": src_text,
            "tgt_text": tgt_text
        }

def causal_mask(size):
    mask = torch.triu(torch.ones(size, size), diagonal=1).type(torch.int)
    return mask == 0


