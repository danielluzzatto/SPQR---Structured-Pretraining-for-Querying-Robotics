import os
import requests
import json
from tqdm import tqdm
import tiktoken
import torch
#------------

DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "hellaswag")

def download_file(url: str, fname: str, chunk_size=1024):
    """Downloads a file from a given URL and saves it locally."""
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))

    with open(fname, 'wb') as file, tqdm(
        desc=fname, total=total, unit="iB", unit_scale=True, unit_divisor=1024
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)

hellaswags = {
    "train": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl",
    "val": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl",
    "test": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_test.jsonl",
}

enc = tiktoken.get_encoding("gpt2")

def download(split):
    """Downloads the HellaSwag dataset for a given split (train, val, test)."""
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    data_url = hellaswags[split]
    data_filename = os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl")
    if not os.path.exists(data_filename):
        print(f"Downloading {data_url} to {data_filename}...")
        download_file(data_url, data_filename)

def render_example(example):
    """Processes a dataset example, encoding context and endings."""
    ctx = example['ctx']
    label = example['label']
    endings = example['endings']

    data = {"label": label, 'ctx_tokens': None, 'endings_tokens': []}
    ctx_tokens = enc.encode(ctx)
    data["ctx_tokens"] = ctx_tokens
    tok_rows, mask_rows = [], []
    
    for end in endings:
        end_tokens = enc.encode(" " + end)
        tok_rows.append(ctx_tokens + end_tokens)
        mask_rows.append([0] * len(ctx_tokens) + [1] * len(end_tokens))
        data['endings_tokens'].append(end_tokens)
    
    max_len = max(len(row) for row in tok_rows)
    tokens = torch.zeros((4, max_len), dtype=torch.long)
    mask = torch.zeros((4, max_len), dtype=torch.long)
    
    for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
        tokens[i, :len(tok_row)] = torch.tensor(tok_row)
        mask[i, :len(tok_row)] = torch.tensor(mask_row)
    
    return data, tokens, mask, label

def iterate_examples(split):
    """Iterates through dataset examples after downloading."""
    download(split)
    with open(os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl")) as f:
        for line in f:
            yield json.loads(line)
