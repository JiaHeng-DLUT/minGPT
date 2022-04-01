"""
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

import math
import random
from turtle import position

import torch
import torch.nn as nn
from torch.nn import functional as F


class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, block_size, **kwargs):
        self.block_size = block_size
        for k,v in kwargs.items():
            setattr(self, k, v)

class GPT1Config(GPTConfig):
    """ GPT-1 like network roughly 125M params """
    n_layer = 24
    n_head = 12
    n_embd = 768

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        num_frame = config.num_tokens
        num_id = config.num_animals
        a = torch.tril(torch.ones(num_frame, num_frame))[:, :, None, None]
        b = torch.ones(1, 1, num_id, num_id)
        mask = (a * b).transpose(1, 2).reshape(num_frame * num_id, -1)[None, None, :, :]
        self.register_buffer("mask", mask)
        print(self.mask.tolist())
        self.n_head = config.n_head

    def forward(self, x, mask, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        m = self.mask * mask[:, None, None, :]
        att = att.masked_fill(m == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, inp):
        (x, mask) = inp
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.mlp(self.ln2(x))
        return (x, mask)

class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config):
        super().__init__()

        # input embedding stem
        self.config = config
        self.tok_emb = nn.Linear(config.input_dim, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.output_dim)
        self.decoder = nn.Sequential(
            nn.Linear(config.output_dim, config.output_dim),
            nn.Tanh(),
            nn.LayerNorm(config.output_dim),
            nn.Linear(config.output_dim, config.input_dim)
        )
        self.frame_decoder = nn.Sequential(
            nn.Linear(config.output_dim, config.output_dim),
            nn.Tanh(),
            nn.LayerNorm(config.output_dim),
            nn.Linear(config.output_dim, config.input_dim * config.num_animals)
        )
        # self.head1 = nn.Linear(config.output_dim, 2, bias=False)
        # self.head2 = nn.Linear(config.output_dim, 2, bias=False)
        # self.head3 = nn.Linear(config.output_dim, 2, bias=False)
        self.bn = nn.BatchNorm2d(config.input_dim)
        self.block_size = config.block_size
        self.apply(self._init_weights)

        print("number of parameters: %e" % sum(p.numel() for p in self.parameters()))

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding, torch.nn.BatchNorm2d)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def forward(self, tokens, pos, mask):
        b = tokens.shape[0]
        t = tokens.shape[1]
        c = tokens.shape[2]
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        # forward the GPT model
        token_embeddings = self.tok_emb(self.bn(tokens.permute(0, 3, 1, 2)).permute(0, 2, 3, 1))
        embeddings = token_embeddings.view(b, t * c, -1)
        x = self.drop(embeddings)
        x2 = x.flip(1)
        x, _ = self.blocks((x, mask))
        x = self.ln_f(x)
        x = self.proj(x)        #(b, num_tokens, output_dim)

        # animal LR
        tokens = tokens.view(b, t * c, -1) * mask[:, :, None]
        pred = self.decoder(x) * mask[:, :, None]
        l_regression = F.mse_loss(pred[:, :-c], tokens[:, c:])
        # animal RL
        x2, _ = self.blocks((x2, mask.flip(-1)))
        x2 = self.ln_f(x2)
        x2 = self.proj(x2)
        pred2 = self.decoder(x2) * mask[:, :, None]
        l_regression_RL = F.mse_loss(pred2[:, :-c], tokens.flip(1)[:, c:])
        # frame LR
        pred_frame_lr = self.frame_decoder(x.view(b, t, c, -1).mean(dim=-2))[:, :-1]
        target_frame_lr = tokens.view(b, t, -1)[:, 1:]
        l_frame_lr = F.mse_loss(pred_frame_lr, target_frame_lr)
        # frame RL
        pred_frame_rl = self.frame_decoder(x2.view(b, t, c, -1).mean(dim=-2))[:, :-1]
        target_frame_rl = tokens.view(b, t, -1).flip(1)[:, 1:]
        l_frame_rl = F.mse_loss(pred_frame_rl, target_frame_rl)

        losses = {
            # 'loss1': loss1,
            # 'loss2': loss2,
            # 'loss3': loss3,
            'l_regression': l_regression,
            'l_regression_RL': l_regression_RL,
            'l_frame_lr': l_frame_lr,
            'l_frame_rl': l_frame_rl,
        }

        x = x.view(b, t, c, -1).mean(dim=-2)
        return x, losses