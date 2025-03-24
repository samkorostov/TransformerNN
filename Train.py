import torch
import torch.nn as nn

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from torch.utils.tensorboard import SummaryWriter

from Dataset import BilingualDataset, causal_mask
from Transformer import *
from config import get_weights_file_path
from config import get_config

import warnings
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import random_split
from torch.utils.data import DataLoader


def greedy_decode(model, src, src_mask, tokenizer_tgt, tokenizer_src, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Precompute encoder output and use for decoding
    memory = model.encode(src, src_mask)
    # Initial input is SOS token
    decoder_input = torch.empty(1,1).fill_(sos_idx).type_as(src).to(device)

    while True:
        if decoder_input.size(1) == max_len:
            break

        decoder_mask = causal_mask(decoder_input.size(1)).type_as(src_mask).to(device)

        # Calculate output
        decoder_output = model.decode(memory, src_mask, decoder_input, decoder_mask)

        # Get next token:
        prob = model.project(decoder_output[:,-1])
        # Select token with max probability
        _, next_token = torch.max(prob, dim=-1)

        # Append to decoder input
        decoder_input = torch.cat([decoder_input, torch.empty(1,1).type_as(src).fill_(next_token.item()).to(device)], dim=-1)

        if next_token == eos_idx:
            break

    return decoder_input.squeeze(0)

def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_state, writer, num_examples=2):
    model.eval()
    count = 0

    src_texts = []
    expected = []
    predicted = []

    console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)

            assert encoder_input.shape[0] == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_tgt, tokenizer_src, max_len, device)

            src_text = batch['src_text'][0]
            tgt_text = batch['tgt_text'][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            src_texts.append(src_text)
            expected.append(tgt_text)
            predicted.append(model_out_text)


            print_msg('-'*console_width)
            print_msg(f"Source: {src_text}")
            print_msg(f"Target: {tgt_text}")
            print_msg(f"Predicted: {model_out_text}")

            if count >= num_examples:
                break


def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]

def get_or_init_tokenizer(config, ds, lang):
    # config['tokenizer_path'] = 'tokenizer_{}.json'
    tokenizer_path = Path(config['tokenizer_path'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[SOS]", "[EOS]", "[PAD]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_dataset(config):
    ds_raw = load_dataset('opus_books', f'{config["lang_src"]}-{config["lang_tgt"]}', split='train')

    # Build Tokenizers
    tokenizer_src = get_or_init_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_init_tokenizer(config, ds_raw, config['lang_tgt'])

    # Split into Train and Validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size

    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids

        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f"Max Length of Source sentence: {max_len_src}")
    print(f"Max Length of Target sentence: {max_len_tgt}")

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=False)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_src_len, vocab_tgt_len):
    model = make_model(vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'], config['d_model'])
    return model

def train_model(config):
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    print(f"Using device: {device}")

    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_dataset(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    # Tensorboard:
    writer = SummaryWriter(config['experiment_name'])

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    initial_epoch = 0
    global_step = 0

    if config['preload'] is not None:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f"Loading model from {model_filename}")

        state = torch.load(model_filename)
        initial_epoch = state['epoch']
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_tgt.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()

        batch_itr = tqdm(train_dataloader, desc=f"Epoch {epoch:02d}", total=len(train_dataloader))
        for batch in batch_itr:
            encoder_input = batch['encoder_input'].to(device) # (Batch, Seq_len)
            decoder_input = batch['decoder_input'].to(device) # (Batch, Seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (Batch, 1, 1, Seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # (Batch, 1, Seq_len, Seq_len)

            # Run tensors through transformer
            memory = model.encode(encoder_input, encoder_mask) # (Batch, Seq_len, d_model)
            decoder_output = model.decode(memory, encoder_mask, decoder_input, decoder_mask) # (Batch, Seq_len, d_model)
            proj_output = model.project(decoder_output) # (Batch, Seq_len, vocab_size)

            # Calculate loss
            label = batch["label"].to(device) # (Batch, Seq_len)

            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1)) # (Batch * Seq_len, vocab_size)
            batch_itr.set_postfix({f"Loss": f"{loss.item():6.3f}"})

            writer.add_scalar("Train loss", loss.item(), global_step)
            writer.flush()

            # Backpropagation
            loss.backward()

            # Update weights
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1

        # Save model at end of epoch
        model_filename = get_weights_file_path(config, f'{epoch:02d}')
        torch.save({
            'epoch': epoch,
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, model_filename)

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)


