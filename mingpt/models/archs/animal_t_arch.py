import math
import torch
import torch.nn as nn
from torch.nn import functional as F

from mingpt.utils import get_root_logger


class CausalSelfAttention(nn.Module):
    """
    From: https://github.com/karpathy/minGPT/blob/3ed14b2cec0dfdad3f4b2831f2b4a86d11aef150/mingpt/model.py#L37

    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, opt):
        super().__init__()
        assert opt['n_embd'] % opt['n_head'] == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(opt['n_embd'], opt['n_embd'])
        self.query = nn.Linear(opt['n_embd'], opt['n_embd'])
        self.value = nn.Linear(opt['n_embd'], opt['n_embd'])
        # regularization
        self.attn_drop = nn.Dropout(opt['attn_pdrop'])
        self.resid_drop = nn.Dropout(opt['resid_pdrop'])
        # output projection
        self.proj = nn.Linear(opt['n_embd'], opt['n_embd'])
        # causal mask to ensure that attention is only applied to the left in the input sequence
        num_frames = opt['num_frames']
        mask = torch.tril(torch.ones(num_frames, num_frames))[None, None, :, :]
        self.register_buffer("mask", mask)
        self.n_head = opt['n_head']

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C //
                             self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C //
                               self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C //
                               self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        # re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):
    """
    From: https://github.com/karpathy/minGPT/blob/3ed14b2cec0dfdad3f4b2831f2b4a86d11aef150/mingpt/model.py#L81

    An unassuming Transformer block.
    """

    def __init__(self, opt):
        super().__init__()
        self.ln1 = nn.LayerNorm(opt['n_embd'])
        self.ln2 = nn.LayerNorm(opt['n_embd'])
        self.attn = CausalSelfAttention(opt)
        self.mlp = nn.Sequential(
            nn.Linear(opt['n_embd'], 4 * opt['n_embd']),
            nn.GELU(),
            nn.Linear(4 * opt['n_embd'], opt['n_embd']),
            nn.Dropout(opt['resid_pdrop']),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class AnimalT(nn.Module):
    """
    Each frame as a token.
    Attention is only applied to the animals in the left frames.
    """

    def __init__(self, opt):
        super().__init__()

        # input embedding stem
        self.opt = opt
        self.tok_emb = nn.Linear(opt['input_dim'], opt['n_embd'])
        self.drop = nn.Dropout(opt['embd_pdrop'])
        # transformer
        self.blocks = nn.Sequential(*[Block(opt)
                                    for _ in range(opt['n_layer'])])
        # decoder head
        self.ln_f = nn.LayerNorm(opt['n_embd'])
        self.proj = nn.Linear(opt['n_embd'], opt['output_dim'])

        self.decoder_frame = nn.Sequential(
            nn.Linear(opt['output_dim'], opt['output_dim']),
            nn.Tanh(),
            nn.LayerNorm(opt['output_dim']),
            nn.Linear(opt['output_dim'], opt['input_dim'])
        )

        self.total_frames = opt['total_frames']
        self.apply(self._init_weights)

        logger = get_root_logger()
        logger.info('Number of parameters: %e' % sum(p.numel()
                    for p in self.parameters()))

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def encode(self, x):
        x = self.drop(x)
        x = self.blocks(x)
        x = self.ln_f(x)
        x = self.proj(x)    # (b, t * c, output_dim)
        return x

    def forward(self, tokens, decode=False):
        """
        Args:
            tokens (torch.Tensor): (b, num_frames, num_animals, input_dim)
            flip (bool, optional): whether to flip or not.
        """
        tokens = tokens.flatten(2)
        b, t, d = tokens.shape

        token_embeddings = self.tok_emb(tokens)
        feats = self.encode(token_embeddings)

        if not decode:
            return feats

        pred = self.decoder_frame(feats)
        pred_delta = pred[:, :-1] - tokens[:, 1:]

        return pred_delta


if __name__ == '__main__':
    opt = {
        'input_dim': 12 * 2,
        'n_embd': 48,
        'n_layer': 12,
        'n_head': 12,
        'output_dim': 128,

        'embd_pdrop': 0.1,
        'resid_pdrop': 0.1,
        'attn_pdrop': 0.1,

        'num_frames': 50,
        'num_animals': 3,
        'total_frames': 1800,

        'decode_animal': True,
        'decode_frame': True,
    }

    model = AnimalT(opt)
    (b, t, c, d) = (32, opt['num_frames'],
                    opt['num_animals'], opt['input_dim'])
    tokens = torch.randn((b, t, c, d))
    masks = torch.ones((b, t, c))
    pos = torch.randperm(b)
    outputs = model(tokens,
                    masks,
                    pos,
                    flip=True,
                    decode_animal=True,
                    decode_frame=True)
    feat_LR = outputs['feat_LR']
    feat_RL = outputs['feat_RL']
    animal_LR = outputs['animal_LR']
    animal_RL = outputs['animal_RL']
    frame_LR = outputs['frame_LR']
    frame_RL = outputs['frame_RL']
    print(feat_LR.shape, feat_RL.shape)
    print(animal_LR.shape, animal_RL.shape)
    print(frame_LR.shape, frame_RL.shape)
    # torch.Size([32, 50, 128]) torch.Size([32, 50, 128])
    # torch.Size([32, 50, 3, 24]) torch.Size([32, 50, 3, 24])
    # torch.Size([32, 50, 72]) torch.Size([32, 50, 72])
