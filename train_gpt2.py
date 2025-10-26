import os
import math
import time
import inspect
import argparse
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from hellaswag import render_example, iterate_examples
# -----------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # 在一个batch中，对全部的head进行query, key, value投影
        # B: batch size
        # T: sequence length
        # C: embedding dimension

        # [B, T, C] -> [B, T, 3 * C]
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # [B, T, 3 * C] -> [B, T, C]
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # n_head: number of heads
        # n_embd: embedding dimension
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        # 输入x的shape是[B, T, C]，最终输出y的shape是[B, T, C]
        # C是n_head * head_size
        # GPT-2 (124M), n_head=12, head_size=64, 所以 n_head * head_size=C=768 即Transformer的通道数
        B, T, C = x.size() 
        
        # 对输入x进行线性变换，得到[B, T, 3 * C]
        qkv = self.c_attn(x)

        # [B, T, 3 * C] -> [B, T, C], [B, T, C], [B, T, C]
        # 将[B, T, 3 * C]沿着第2维（feature维）按照每份n_embd的大小分成3份，也就是q、k、v三个张量。
        q, k, v = qkv.split(self.n_embd, dim=2)

        # 将qkv分别转换为[B, T, n_head, head_size]的形状，再转置得到[B, n_head, T, head_size]
        # 之所以要转置，是会把batch与时间步维度放到前面，方便后续的计算。
        # [B, T, C] -> [B, n_head, T, head_size]
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # 使用flash attention计算注意力，得到[B, n_head, T, head_size]
        # is_causal=True表示是因果注意力，即只能看到前面的token，不能看到后面的token。
        # 返回的y的shape是[B, n_head, T, head_size]
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention

        # 将[B, n_head, T, head_size]转置回[B, T, C]，再拼接起来
        # 其中contiguous()是为了确保tensor在内存中是连续的，因为transpose会破坏连续性，所以需要重新排列内存，方便后续的计算。
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # 对输出y进行线性变换，shape不变，还是[B, T, C]
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        # 全连接层，将[B, T, C] -> [B, T, 4 * C]
        # 在GPT-2中，MLP的中间层是4倍于输入层的大小。
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        # GELU激活函数
        self.gelu    = nn.GELU(approximate='tanh')
        # 全连接层，将[B, T, 4 * C] -> [B, T, C]
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        # 初始化权重，使得输出方差与输入方差一致
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        # 输入x的shape是[B, T, C]，最终输出y的shape是[B, T, C]
        # C是n_embd，即Transformer的通道数
        # C=n_head * head_size
        # GPT-2 (124M), n_head=12, head_size=64, 所以 n_head * head_size=C=768 即Transformer的通道数
        x = self.c_fc(x)
        # 对x进行线性变换，得到[B, T, 4 * C]
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        # LayerNorm, 将[B, T, C] -> [B, T, C]，不改变shape
        self.ln_1 = nn.LayerNorm(config.n_embd)
        # CausalSelfAttention，将[B, T, C] -> [B, T, C]，不改变shape
        self.attn = CausalSelfAttention(config)
        # LayerNorm, 将[B, T, C] -> [B, T, C]，不改变shape
        self.ln_2 = nn.LayerNorm(config.n_embd)
        # MLP，将[B, T, C] -> [B, T, C]，不改变shape
        self.mlp = MLP(config)
    
    def forward(self, x):
        # 将x与self.attn(self.ln_1(x))相加，得到[B, T, C]
        x = x + self.attn(self.ln_1(x))
        # 将x与self.mlp(self.ln_2(x))相加，得到[B, T, C]
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension

class GPT(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            # nn.Embedding: 将一个整数（词的索引）映射为一个向量（词的嵌入表示）

            
            # config.vocab_size: 词表大小
            # config.block_size: 最大序列长度
            # config.n_embd: 词嵌入维度

            # token embedding
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            # position embedding
            wpe = nn.Embedding(config.block_size, config.n_embd),
            # blocks
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            # final layer norm
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        # 最后将Transformer的输出映射到词表空间，将[B, T, C] -> [B, T, vocab_size]
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        # 这里进行权重共享，将token embedding的权重与最后输出层的权重共享，这样可以节省内存，并且使得模型更加稳定。
        # GPT-2的实现中，token embedding的权重与最后输出层的权重是共享的。
        # 我们也预期这么做是合理的，因为一个训练充分的模型，从id到embeding和从embedding到logits的映射应该是一样的。
        self.transformer.wte.weight = self.lm_head.weight

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # 初始化权重，使得输出方差与输入方差一致
            std = 0.02
            # 如果模块有NANOGPT_SCALE_INIT属性，则使用更小的初始化方差，这是为了防止初始化方差过大，导致模型训练不稳定。
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # 初始化embedding的权重，使得输出方差与输入方差一致
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # 输入idx的shape是[B, T]，其中B是batch size，T是序列长度
        B, T = idx.size()
        # 确保序列长度不超过最大序列长度
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # 计算位置编码
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # shape (B, T, n_embd)
        # 将token embedding和位置编码element-wise相加，得到[B, T, n_embd]
        x = tok_emb + pos_emb
        # 前向传播Transformer的每个block，得到[B, T, n_embd]
        for block in self.transformer.h:
            x = block(x)
        # 最后将Transformer的输出进行LayerNorm，得到[B, T, n_embd]
        x = self.transformer.ln_f(x)
        # 将Transformer的输出映射到词表空间，得到[B, T, vocab_size]
        logits = self.lm_head(x) # (B, T, vocab_size)
        # 如果targets不为None，则计算交叉熵损失，这个targets是真实的目标序列，用于计算损失。
        # 之所以targets可能为None，是因为在预处理数据时，遇到最后一个token时，没有目标序列，所以targets为None。
        # 另外，在推理时，targets也为None。
        loss = None
        if targets is not None:
            # 这里用的是原始logits，而不是softmax结果，因为F.cross_entropy会自动内部计算softmax，所以不需要再手动计算。
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
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
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # 获取所有需要梯度更新的参数
        # named_parameters()返回一个生成器
        # 生成器中的每个元素是一个元组
        # 元组中第一个元素是参数的名称
        # 第二个元素是参数本身。
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # 创建优化组。任何2D的参数都会进行权重衰减，否则不进行权重衰减。
        # 即所有矩阵型的weight tensors in matmuls + embeddings都会进行权重衰减，所有偏置和LayerNorm的参数都不会进行权重衰减。
        # 2D的参数包括：全连接层的权重和注意力机制的权重等。
        # 1D的参数包括：偏置和LayerNorm的参数等。
        # 之所以要进行权重衰减，是因为我们希望模型的参数不要过大，否则会导致模型过拟合。
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        
        # 创建优化组，其中decay_params是进行权重衰减的参数，nodecay_params是不进行权重衰减的参数。
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        # 计算进行权重衰减的参数和未进行权重衰减的参数的数量
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        # 打印进行权重衰减的参数和未进行权重衰减的参数的数量
        if master_process:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        
        # 创建AdamW优化器，并使用fused版本（如果可用）
        # fused版本是PyTorch官方提供的一种优化，可以加速优化器的计算。
        # 如果fused版本可用，并且设备类型为cuda，则使用fused版本。
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        
        if master_process:
            print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

# -----------------------------------------------------------------------------
import tiktoken
import numpy as np
# 加载tokens
# 输入：文件名
# 输出：tensor，shape是[num_tokens]
def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32) # 将numpy数组转换为int32类型
    ptt = torch.tensor(npt, dtype=torch.long) # 将numpy数组转换为torch.long类型
    return ptt

# 数据加载器
# 输入：batch size, 序列长度, 进程排名, 进程数量, 数据集类型
# 输出：x, y
# x是输入序列，y是目标序列
class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        # get the shard filenames
        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y

# -----------------------------------------------------------------------------
# helper function for HellaSwag eval
# takes tokens, mask, and logits, returns the index of the completion with the lowest loss

def get_most_likely_row(tokens, mask, logits):
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm

# -----------------------------------------------------------------------------
# simple launch:
# python train_gpt2.py
# DDP launch for e.g. 8 GPUs:
# torchrun --standalone --nproc_per_node=8 train_gpt2.py

# run the training loop
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# set up DDP (distributed data parallel).
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

# added after video, pytorch can be serious about it's device vs. device_type distinction
device_type = "cuda" if device.startswith("cuda") else "cpu"

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

enc = tiktoken.get_encoding("gpt2")

total_batch_size = 524288 # 2**19, ~0.5M, in number of tokens
B = 16 # micro batch size
T = 1024 # sequence length
assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train")
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val")

torch.set_float32_matmul_precision('high')

# create model
model = GPT(GPTConfig(vocab_size=50304))
# model = GPT.from_pretrained("gpt2") # or init from OpenAI GPT-2

# 检查点恢复（在DDP包装之前）
parser = argparse.ArgumentParser()
parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint (e.g., log/model_15000.pt)")
parser.add_argument("--inference", type=str, default=None, help="Inference mode: load checkpoint and generate text (e.g., log/model_15000.pt)")
args = parser.parse_args()

start_step = 0
if args.resume:
    ckpt = torch.load(args.resume, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    start_step = ckpt['step'] + 1
    if master_process:
        print(f"✓ 从 {args.resume} 恢复训练 (第 {start_step} 步开始)")

# 推理模式
if args.inference:
    if master_process:
        print(f"✓ 推理模式：加载模型权重 {args.inference}")
    ckpt = torch.load(args.inference, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    model.to(device)
    model.eval()
    
    # 生成文本5次
    num_return_sequences = 5
    max_length = 32
    prompt = "Hello, I'm a language model,"
    tokens = enc.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
    xgen = tokens.to(device)
    sample_rng = torch.Generator(device=device)
    sample_rng.manual_seed(42)
    
    if master_process:
        print(f"\n{'='*60}")
        print(f"提示词: {prompt}")
        print(f"{'='*60}\n")
    
    while xgen.size(1) < max_length:
        with torch.no_grad():
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, loss = model(xgen)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            ix = torch.multinomial(topk_probs, 1, generator=sample_rng)
            xcol = torch.gather(topk_indices, -1, ix)
            xgen = torch.cat((xgen, xcol), dim=1)
    
    if master_process:
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"生成 {i+1}: {decoded}\n")
    
    # 推理完成后退出
    import sys
    sys.exit(0)

model.to(device)
use_compile = False # torch.compile interferes with HellaSwag eval and Generation. TODO fix
if use_compile:
    model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model # always contains the "raw" unwrapped model

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715
max_steps = 19073 # 19,073 steps is ~1 epoch, if data is 10B tokens and batch size 0.5M tokens
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

# optimize!
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device_type)

# create the log directory we will write checkpoints to and log to
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
with open(log_file, "w") as f: # open for writing to clear the file
    pass

for step in range(start_step, max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)

    # once in a while evaluate our validation loss
    if step % 250 == 0 or last_step:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            print(f"validation loss: {val_loss_accum.item():.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} val {val_loss_accum.item():.4f}\n")
            if step > 0 and (step % 5000 == 0 or last_step):
                # optionally write model checkpoints
                checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'config': raw_model.config,
                    'step': step,
                    'val_loss': val_loss_accum.item()
                }
                # you might also want to add optimizer.state_dict() and
                # rng seeds etc., if you wanted to more exactly resume training
                torch.save(checkpoint, checkpoint_path)

    # once in a while evaluate hellaswag
    if (step % 250 == 0 or last_step) and (not use_compile):
        num_correct_norm = 0
        num_total = 0
        for i, example in enumerate(iterate_examples("val")):
            # only process examples where i % ddp_world_size == ddp_rank
            if i % ddp_world_size != ddp_rank:
                continue
            # render the example into tokens and labels
            _, tokens, mask, label = render_example(example)
            tokens = tokens.to(device)
            mask = mask.to(device)
            # get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(tokens)
                pred_norm = get_most_likely_row(tokens, mask, logits)
            num_total += 1
            num_correct_norm += int(pred_norm == label)
        # reduce the stats across all processes
        if ddp:
            num_total = torch.tensor(num_total, dtype=torch.long, device=device)
            num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
            dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
            num_total = num_total.item()
            num_correct_norm = num_correct_norm.item()
        acc_norm = num_correct_norm / num_total
        if master_process:
            print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} hella {acc_norm:.4f}\n")

    # once in a while generate from the model (except step 0, which is noise)
    if ((step > 0 and step % 250 == 0) or last_step) and (not use_compile):
        model.eval()
        num_return_sequences = 4
        max_length = 32
        tokens = enc.encode("Hello, I'm a language model,")
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42 + ddp_rank)
        while xgen.size(1) < max_length:
            # forward the model to get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(xgen) # (B, T, vocab_size)
                # take the logits at the last position
                logits = logits[:, -1, :] # (B, vocab_size)
                # get the probabilities
                probs = F.softmax(logits, dim=-1)
                # do top-k sampling of 50 (huggingface pipeline default)
                # topk_probs here becomes (5, 50), topk_indices is (5, 50)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                # select a token from the top-k probabilities
                # note: multinomial does not demand the input to sum to 1
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
                # gather the corresponding indices
                xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
                # append to the sequence
                xgen = torch.cat((xgen, xcol), dim=1)
        # print the generated text
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"rank {ddp_rank} sample {i}: {decoded}")

    # do one step of the optimization
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        # added after video, this field is also used by the forward pass.
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        # we have to scale the loss to account for gradient accumulation,
        # because the gradients just add on each successive backward().
        # addition of gradients corresponds to a SUM in the objective, but
        # instead of a SUM we want MEAN. Scale the loss here so it comes out right
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # determine and set the learning rate for this iteration
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    if device_type == "cuda":
        torch.cuda.synchronize() # wait for the GPU to finish work
    t1 = time.time()
    dt = t1 - t0 # time difference in seconds
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt
    if master_process:
        print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
        with open(log_file, "a") as f:
            f.write(f"{step} train {loss_accum.item():.6f}\n")

if ddp:
    destroy_process_group()
