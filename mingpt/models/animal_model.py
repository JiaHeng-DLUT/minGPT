import importlib
import torch
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
from torch.cuda.amp import autocast as autocast, GradScaler
from tqdm import tqdm

from mingpt.models.archs import define_network
from mingpt.models.base_model import BaseModel
from mingpt.utils import get_root_logger
from mingpt.eval import Evaluator

loss_module = importlib.import_module('mingpt.models.losses')


class AnimalModel(BaseModel):
    """Each animal in each frame as a token.
    num_tokens = num_animals * num_frames
    """

    def __init__(self, opt):
        super(AnimalModel, self).__init__(opt)

        self.evaluator = Evaluator(opt['val']['val_opt'])
        self.metrics = []

        # define network
        self.net = define_network(deepcopy(opt['network']))
        self.net = self.model_to_device(self.net)
        self.print_network(self.net)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network', None)
        if load_path is not None:
            self.load_network(self.net, load_path,
                              self.opt['path'].get('strict_load', True))

        if self.is_train:
            self.init_training_settings()

        self.scaler = GradScaler()

    def init_training_settings(self):
        self.net.train()
        train_opt = self.opt['train']

        # define losses
        if train_opt.get('animal_opt'):
            animal_type = train_opt['animal_opt'].pop('type')
            cri_animal_cls = getattr(loss_module, animal_type)
            self.cri_animal = cri_animal_cls(
                **train_opt['animal_opt']).to(self.device)
        else:
            self.cri_animal = None

        if train_opt.get('frame_opt'):
            frame_type = train_opt['frame_opt'].pop('type')
            cri_frame_cls = getattr(loss_module, frame_type)
            self.cri_frame = cri_frame_cls(
                **train_opt['frame_opt']).to(self.device)
        else:
            self.cri_frame = None

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        """
        From: https://github.com/karpathy/minGPT/blob/3ed14b2cec0dfdad3f4b2831f2b4a86d11aef150/mingpt/model.py#L136

        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (
            torch.nn.LayerNorm, torch.nn.Embedding, torch.nn.BatchNorm2d)
        for mn, m in self.net.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.net.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(
            inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params))

        train_opt = self.opt['train']

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(
                list(decay))], "weight_decay": train_opt['optim']['weight_decay']},
            {"params": [param_dict[pn]
                        for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]

        optim_type = train_opt['optim'].pop('type')
        if optim_type == 'Adam':
            self.optimizer = torch.optim.Adam(
                optim_groups, **train_opt['optim'])
        if optim_type == 'AdamW':
            self.optimizer = torch.optim.AdamW(
                optim_groups, **train_opt['optim'])
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer)

    def feed_data(self, data):
        self.tokens = data['keypoints'].to(self.device)
        self.mask = data['mask'].to(self.device).long()
        self.pos = data['pos']
        if 'labels' in data:
            self.labels = data['labels'].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer.zero_grad()
        decode_animal = (self.cri_animal is not None)
        decode_frame = (self.cri_frame is not None)
        flip = self.opt['train']['flip']
        with autocast():
            self.output = self.net(self.tokens, self.mask, self.pos, flip=flip,
                                   decode_animal=decode_animal, decode_frame=decode_frame)

            l_total = 0
            loss_dict = OrderedDict()
            # animal reconstrction loss
            if self.cri_animal:
                l_animal_LR = self.cri_animal(
                    self.output['animal_LR'], torch.zeros_like(self.output['animal_LR']))
                l_total += l_animal_LR
                loss_dict['l_animal_LR'] = l_animal_LR
                if flip:
                    l_animal_RL = self.cri_animal(
                        self.output['animal_RL'], torch.zeros_like(self.output['animal_RL']))
                    l_total += l_animal_RL
                    loss_dict['l_animal_RL'] = l_animal_RL

            if self.cri_frame:
                l_frame_LR = self.cri_frame(
                    self.output['frame_LR'], torch.zeros_like(self.output['frame_LR']))
                l_total += l_frame_LR
                loss_dict['l_frame_LR'] = l_frame_LR
                if flip:
                    l_frame_RL = self.cri_frame(
                        self.output['frame_RL'], torch.zeros_like(self.output['frame_RL']))
                    l_total += l_frame_RL
                    loss_dict['l_frame_RL'] = l_frame_RL

        self.scaler.scale(l_total).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(
            self.net.parameters(), self.opt['train']['grad_norm_clip'])
        self.scaler.step(self.optimizer)
        self.scaler.update()

        self.log_dict = self.reduce_loss_dict(loss_dict)

    def test(self):
        self.net.eval()
        with torch.no_grad():
            self.output = self.net(self.tokens, self.mask, self.pos)
        self.net.train()

    def dist_validation(self, dataloader, current_iter, tb_logger):
        logger = get_root_logger()
        logger.info('Only support single GPU validation.')
        self.nondist_validation(dataloader, current_iter, tb_logger)

    def nondist_validation(self, dataloader, current_iter, tb_logger):
        dataset_name = dataloader.dataset.opt['name']

        feats = []
        labels = []
        for val_data in tqdm(dataloader):
            self.feed_data(val_data)
            self.test()

            # del self.lq
            # del self.output
            # torch.cuda.empty_cache()

            feat = self.output['feat_LR']
            feat = feat.view(-1, feat.shape[-1])
            feats.append(feat)
            label = self.labels
            label = label.reshape(-1, label.shape[-1])
            labels.append(label)

        feats = torch.cat(feats, dim=0)
        labels = torch.cat(labels, dim=0)
        assert feats.shape[0] == labels.shape[0]
        metric = self.evaluator.eval(feats, labels)

        self.metrics.append(metric)
        metrics = torch.stack(self.metrics, dim=0)
        # https://www.cnblogs.com/wanghui-garcia/p/12982732.html
        rank = metrics.sort(dim=0, descending=True)[1].sort(dim=0)[1]
        rank_mean = rank.float().mean(dim=1)
        logger = get_root_logger()
        logger.info(f'{metrics}')
        logger.info(f'{rank}')
        logger.info(f'{rank_mean}')
        # if tb_logger:
        #     tb_logger.add_scalar('metric', metric, current_iter)

    def save(self, epoch, current_iter):
        self.save_network(self.net, 'net', current_iter)
        # self.save_training_state(epoch, current_iter)
