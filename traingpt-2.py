from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math, sys, time, inspect, os
import tiktoken
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
from datetime import datetime

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head ==0

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size,config.block_size))
                             .view(1,1, config.block_size, config.block_size))
        
    def forward(self,x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim = 2)

        k = k.view(B, T, self.n_head, C//self.n_head).transpose(1,2) # (B, T, nh, hs)
        q = q.view(B, T, self.n_head, C//self.n_head).transpose(1,2) # (B, T, nh, hs)
        v = v.view(B, T, self.n_head, C//self.n_head).transpose(1,2) # (B, T, nh, hs)

        # att = (q @ k.transpose(-2,-1)) * (1.0/math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float("-inf"))
        # att = F.softmax(att, dim = -1) # changed here to -1
        # y = att @ v
        y = F.scaled_dot_product_attention(q, k, v, is_causal= True) # Flash attention

        y = y.transpose(1,2).contiguous().view(    B,T,C)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh") # for reproducibiliy with gpt2 otherwise better without passing anything in GeLU
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    # bias: bool = True

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), # weight of the tokens' embedding
            wpe = nn.Embedding(config.block_size, config.n_embd), # weight of the position embedding
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd)
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias = False)
        
        self.transformer.wte.weight = self.lm_head.weight  # these weights are supposed to be the same

        # initialize parameters 
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer)**-0.5
            torch.nn.init.normal_(module.weight, mean= 0.0, std= std)
            if module.bias is not None:
                torch.nn.init.normal_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)

    def forward(self, idx, targets = None):
        #  idx.shape (B,T)
        B,T = idx.size()
        assert T<=self.config.block_size, f"Cannote forward sequence of length {T} when block size is {self.config.block_size}"

        pos = torch.arange(0,T, dtype=torch.long, device = idx.device) # shape T
        pos_emb = self.transformer.wpe(pos)  # shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # shape (B, T, n_embd)

        x = pos_emb + tok_emb # broadcasting, the token embedding is replicated B times
        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) # cross entropy does not like multiple dimensional input so we flatten in (B*T,vocab_size) 
        return logits, loss


    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
    
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params --> the one we currently use
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints

        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but here linear layer
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, device):
        param_dict = {pn:p for pn,p in  self.named_parameters()}
        param_dict = {pn:p for pn,p in param_dict.items() if p.requires_grad}
        # not all params need decay --> only the multidimensional ones, like the ones which go through matrix mul and not biases
        decay_params = [p for n,p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n,p in param_dict.items() if p.dim() < 2]
        optim_groups = [{"params": decay_params, "weight_decay": weight_decay},
                        {"params": nodecay_params, "weight_decay": 0.0}]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # use fused for kernel fusion --> runs faster
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        print(f'using fused AdamW:{use_fused}')
        optimizer =torch.optim.AdamW(optim_groups, lr = learning_rate, betas = (0.9, 0.95), eps = 1e-8, fused= use_fused)
        return optimizer

def load_tokens(filename):
    npt = np.load(filename)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt


class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        super().__init__()
        self.B = B
        self.T = T
        self.num_processes = num_processes
        self.process_rank = process_rank
        assert split in {"train", "val"}
        data_root ="edu_fineweb"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root,s) for s in shards]
        self.shards = shards

        assert len(shards)>0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards in split {split}")

        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])

        self.current_position = self.B * self.T * self.process_rank
        self.reset()
    
    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = B * T * self.num_processes

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position: self.current_position + B * T + 1]
        x = (buf[:-1]).view(B,T)
        y = (buf[1:]).view(B,T)

        self.current_position += B * T * self.num_processes
        
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = self.B * self.T * self.process_rank
        return x, y

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"

ddp = int(os.environ.get('RANK', -1)) !=-1
if ddp:
    assert torch.cuda.is_available()
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda: {ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank = 0
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    device = "cpu"
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = 'mps'
    print(f"using device: {device}")
print(f"ddp world size: {ddp_world_size}")


torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

total_batch_size = 8192 # 524288 # 2**19
B = 4 # 16 or 32
T = 1024
assert total_batch_size % (B * T * ddp_world_size) == 0
grad_accum_steps = total_batch_size // (B*T*ddp_world_size)
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")


train_loader =DataLoaderLite(B = B, T = T, process_rank = ddp_rank, num_processes = ddp_world_size, split= "train")
val_loader =DataLoaderLite(B = B, T = T, process_rank = ddp_rank, num_processes = ddp_world_size, split= "val")


torch.set_float32_matmul_precision('high') # using less precision but still affected by memory bounds


model = GPT(GPTConfig(vocab_size= 50304)) # want a power of 2 for performace optimization
model.to(device)
if ddp:
    model = DDP(model, device_ids =[ddp_local_rank])
raw_model = model.module if ddp else model

# Compile Model. Make sure it does not crash
# model = torch.compile(model) 


max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10 # 715
max_steps = 201 # 19073

def get_lr(it): #  lr from gpt 3 cause the one in 2 is not available
    if it < warmup_steps:
        return max_lr * (it+1)/warmup_steps
    if it>max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

# logging results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = f"logs/run_{timestamp}"
os.makedirs(log_dir, exist_ok=True)

train_losses = []
val_losses = []
lrs = []
grad_norms = []
tokens_per_sec_list = []
generated_samples = []

optimizer = raw_model.configure_optimizers(weight_decay = 0.1, learning_rate = 6e-4, device = device)
for step in range(max_steps):
    t0 = time.time()
    # every 100 steps evaluate the loss in val step
    if step % 100 == 0:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x,y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device, dtype = torch.bfloat16):
                    logits, loss = model(x,y)
                loss = loss/val_loss_steps
                val_loss_accum +=loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum, op = dist.ReduceOp.AVG)
        if master_process:
            val_losses.append(val_loss_accum.item())
            np.save(os.path.join(log_dir, "val_losses.npy"), np.array(val_losses))
            print(f"validation loss: {val_loss_accum.item():.4f}")
    
    # check results of the model after validation, not after first step cause it is noisy
    if step > 0 and step % 100 == 0:
        model.eval()
        num_return_sequence = 4
        max_length = 32
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode("Hello, I'm a language model")
        tokens = torch.tensor(tokens, dtype= torch.long) # shape 8
        tokens = tokens.unsqueeze(0).repeat(num_return_sequence, 1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device)
        sample_rng.manual_seed(42 + ddp_rank)
        while xgen.size(1) < max_length:
            with torch.no_grad():
                logits, loss = model(xgen) # (B,T, vocab_size)
                logits = logits[:,-1,:] # (B, vocab_size)
                probs = F.softmax(logits)
                topk_probs, topk_indices = torch.topk(probs, 50, dim = -1)
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng)
                xcol = torch.gather(topk_indices, -1, ix)
                xgen = torch.cat((xgen, xcol), dim = 1)
                for i in range(num_return_sequence):
                    tokens = xgen[i, :max_length].tolist()
                    decoded = enc.decode(tokens)
                    print(f"rank {ddp_rank}, sample: {i}: {decoded}")

    # training
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        # uncomment when using cuda
        # with torch.autocast(device_type= device, dtype= torch.bfloat16):
        #     logits, loss = model(x, y)
        logits, loss = model(x, y)
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps -1)
        loss.backward()
    if ddp:
        dist.all_reduce(loss_accum, op = dist.Reduce0p.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # avoid getting high loss for particular batches --> avoid optim. shocks
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    optimizer.step()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.time()
    dt = t1-t0
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = (tokens_processed) / (t1-t0)
    if master_process:
        print(f"step {step} | loss: {loss_accum:.5f} | lr: {lr:.4e} | dt: {dt:.2f} sec | norm: {norm:.4f} | tok/sec: {tokens_per_sec:.5f}")
        
        train_losses.append(loss_accum.item())
        np.save(os.path.join(log_dir, "train_losses.npy"), np.array(train_losses))
        lrs.append(lr)
        np.save(os.path.join(log_dir, "lr.npy"), np.array(lrs))
        grad_norms.append(norm.item())
        np.save(os.path.join(log_dir, "grad_norms.npy"), np.array(grad_norms))
        tokens_per_sec_list.append(tokens_per_sec)
        np.save(os.path.join(log_dir, "tokens_per_sec.npy"), np.array(tokens_per_sec_list)) 


if ddp:
    destroy_process_group()
sys.exit(0)




































# print(loss)
# sys.exit(0)


# enc = tiktoken.get_encoding('gpt2')
# tokens = enc.encode("Hello, I'm a language model")

# x = tokens.to(device)

# torch.manual_seed(42)
# while x.size(1) < max_length:
#     logits = model(x)
#     logits = logits[:,-1,:]
#     probs = F.softmax(logits, dim=1)
#     topkprobs, topkindices = torch.topk(probs, 50, dim=-1)
#     ix = torch.multinomial(topkprobs, 1)
#     xcol = torch.gather(topkindices, -1, ix)
#     x = torch.cat((x, xcol), dim = 1)

# for i in range(num_return_sequence):
#     tokens = x[i, :max_length].tolist()
#     decoded = enc.decode(tokens)
#     print(">", decoded)