
import torch
import torch.nn as nn
from torch.utils.data import Dataset



class BilingualDataset(Dataset):

    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()
        self.seq_len = seq_len

        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
    src_target_pair = self.ds[idx]
    src_text = src_target_pair['translation'][self.src_lang]
    tgt_text = src_target_pair['translation'][self.tgt_lang]

    # Transform the text into tokens
    enc_input_tokens = self.tokenizer_src.encode(src_text).ids
    dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

    print(f"Original lengths — enc: {len(enc_input_tokens)}, dec: {len(dec_input_tokens)}")


    # Truncate BEFORE computing padding
    enc_input_tokens = enc_input_tokens[:self.seq_len - 2]  # <s> and </s>
    dec_input_tokens = dec_input_tokens[:self.seq_len - 1]  # <s> only

    # Recompute padding AFTER truncation
    enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2
    dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

    # Final safety check — should never hit this now, but good to keep
    if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
        raise ValueError(f"Still too long after truncation: enc={len(enc_input_tokens)}, dec={len(dec_input_tokens)}")

    # Add <s> and </s> tokens and pad
    encoder_input = torch.cat(
        [
            self.sos_token,
            torch.tensor(enc_input_tokens, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64),
        ],
        dim=0,
    )

    decoder_input = torch.cat(
        [
            self.sos_token,
            torch.tensor(dec_input_tokens, dtype=torch.int64),
            torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
        ],
        dim=0,
    )

    label = torch.cat(
        [
            torch.tensor(dec_input_tokens, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
        ],
        dim=0,
    )

    # Final sanity check
    assert encoder_input.size(0) == self.seq_len, f"Encoder input wrong size: {encoder_input.size(0)}"
    assert decoder_input.size(0) == self.seq_len, f"Decoder input wrong size: {decoder_input.size(0)}"
    assert label.size(0) == self.seq_len, f"Label wrong size: {label.size(0)}"

    return {
        "encoder_input": encoder_input,
        "decoder_input": decoder_input,
        "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),
        "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),
        "label": label,
        "src_text": src_text,
        "tgt_text": tgt_text,
    }

    
def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0
