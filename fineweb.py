import tiktoken
from datasets import load_dataset
from tqdm import tqdm
import os
import numpy as np
import multiprocessing as mp
# ------------------------------------------ 

local_dir = "edu_fineweb"
remote_name = 'sample-10BT'
shard_size = int(1e8)

DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok= True)

# download dataset
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens["<|endoftext|>"]

def tokenize(doc):
    # tokenize document --> return numpy array of uint16 tokens
    tokens = [eot] # last token and also need to go first token in document
    tokens.extend(enc.encode(doc["text"], allowed_special={"<|endoftext|>"}))  # Allow special token
    tokens_np = np.array(tokens)
    assert (0<= tokens_np).all() and (tokens_np < 2**16).all(), "token dict too large or empty"
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    
    return tokens_np_uint16

def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)


if __name__ == "__main__":
    nprocs = max(1, os.cpu_count()//2) # count number of cpus in system
    fw = load_dataset("HuggingFaceFW/fineweb-edu", name= remote_name, split= 'train')

    with mp.Pool(nprocs) as pool:
        shard_index = 0
        all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
        tokens_count = 0

        progress_bar = None
        for tokens in pool.imap(tokenize, fw, chunksize = 16):
            if tokens_count + len(tokens) < shard_size:
                # there is enough space --> append tokens
                all_tokens_np[tokens_count:tokens_count+len(tokens)] = tokens
                tokens_count += len(tokens)
                if progress_bar is None:
                    progress_bar = tqdm(total= shard_size, unit= 'tokens', desc= f"Shard {shard_index}")
                progress_bar.update(len(tokens))
            else:
                # create new shard
                split = "val" if shard_index == 0 else 'train'
                filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}.npy") 
                remainder = shard_size - tokens_count
                progress_bar.update(remainder)
                all_tokens_np[tokens_count:tokens_count+remainder] = tokens[:remainder]

                write_datafile(filename, all_tokens_np)
                shard_index +=1
                progress_bar = None

                all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
                tokens_count = len(tokens) - remainder
        # write any remaining token as last shard
        if tokens_count != 0:
            split = "val" if shard_index == 0 else 'train'
            filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}.npy")
            print("last shard")
            write_datafile(filename, all_tokens_np[:tokens_count])
                