import copy
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from mingpt.utils import get_time_str, get_root_logger


class Head(nn.Module):
    def __init__(self, opt):
        super().__init__()
        input_dim = opt['input_dim']
        output_dim = opt['output_dim']
        self.fc = nn.Linear(input_dim, output_dim, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x):
        return self.fc(x)


def create_dataloader(feat, label, index):
    feat = feat[index]
    label = label[index]
    data = {
        'feat': feat,
        'label': label,
    }
    return data


def train(seed, subtask, data, model, optimizer, pos_weight):
    logger = get_root_logger()
    train_data = data['train']
    val_data = data['val']
    # test_data = data['test']

    best_metric = 0
    best_state_dict = copy.deepcopy(model.state_dict())
    for e in range(20):
        model.train()
        feat = train_data['feat']
        label = train_data['label'][:, subtask]
        logit = model(feat).squeeze(-1)
        weight = (label.shape[0] - label.sum()) / label.sum()
        loss = F.binary_cross_entropy_with_logits(
            logit, label, pos_weight=weight)
        model.zero_grad()
        loss.backward()
        optimizer.step()
        (acc, P, R, f1, mP) = val_test(subtask, val_data, model)
        # logger.info(
        #     f'Epoch: {e + 1}, ACC: {acc:.5f}, P: {P:.5f}, R: {R:.5f}, F1: {f1:.5f}, mP: {mP:.5f}')
        metric = mP
        if metric > best_metric:
            best_metric = metric
            best_state_dict = copy.deepcopy(model.state_dict())

    return (best_metric, best_state_dict)


@torch.no_grad()
def val_test(subtask, data, model):
    model.eval()
    feat = data['feat']
    label = data['label'][:, subtask]
    logit = model(feat).squeeze(-1)
    metrics = cal_metrics(logit, label)
    return metrics


def cal_metrics(logit, label):
    pred = (logit > 0)
    TP = ((label == 1) & (pred == 1)).sum().item()
    FN = ((label == 1) & (pred == 0)).sum().item()
    FP = ((label == 0) & (pred == 1)).sum().item()
    TN = ((label == 0) & (pred == 0)).sum().item()

    acc = P = R = f1 = P0 = 0
    if (TP + FN + FP + TN) != 0:
        acc = (TP + TN) / (TP + FN + FP + TN)
    if (TP + FP) != 0:
        P = TP / (TP + FP)
    if (TP + TN) != 0:
        R = TP / (TP + TN)
    if (P + R) != 0:
        f1 = 2 * P * R / (P + R)
    if (TN + FN) != 0:
        P0 = TN / (TN + FN)
    mP = (P + P0) / 2.
    return (acc, P, R, f1, mP)


class Evaluator:
    def __init__(self, opt):
        self.opt = opt
        self.num_seeds = opt['num_seeds']
        self.num_subtasks = opt['num_subtasks']
        self.lr_list = opt['lr_list']
        self.index_list = []
        self.model = Head(opt).to(0)
        self.init_state_dict = copy.deepcopy(self.model.state_dict())

    def eval(self, feat, label):
        """num_seeds * num_subtasks * num_lrs = 3 * 2 * 9
        """
        logger = get_root_logger()
        num = feat.shape[0]
        if len(self.index_list) == 0:
            for i in range(self.num_seeds):
                index = list(range(num))
                random.shuffle(index)
                self.index_list.append(index)
        logger.info(f'Start evaluation')
        logger.info(f'Index: {[index[:5] for index in self.index_list]}')
        result = torch.zeros((self.num_seeds, self.num_subtasks))
        # 1. loop seeds
        for seed, index in enumerate(self.index_list):
            # 2. split train, val and test
            num_train = int(num * 0.6)
            num_val = int(num * 0.2)
            train_index = index[: num_train]
            val_index = index[num_train: num_train + num_val]
            test_index = index[num_train + num_val:]
            data = {
                'train': create_dataloader(feat, label, train_index),
                'val': create_dataloader(feat, label, val_index),
                'test': create_dataloader(feat, label, test_index),
            }
            # 3. loop subtasks
            for subtask in range(self.num_subtasks):
                # 4. loop lr
                best_metric = 0.
                best_state_dict = self.init_state_dict
                for lr in self.lr_list:
                    # 5. train subtask model
                    # create model
                    self.model.load_state_dict(self.init_state_dict)
                    # create optimizer
                    optimizer = optim.SGD(
                        self.model.parameters(), lr, momentum=0.9)
                    # create pos_weight
                    pos_weight = torch.Tensor(
                        [self.opt['pos_weight'][subtask]]).to(0)
                    (metric, state_dict) = train(
                        seed, subtask, data, self.model, optimizer, pos_weight)
                    if metric > best_metric:
                        best_metric = metric
                        best_state_dict = state_dict
                    logger.info(
                        f'lr: {lr:e}, metric: {metric:.5f}, best_metric: {best_metric:.5f}')
                # 6. test
                self.model.load_state_dict(best_state_dict)
                (acc, P, R, f1, mP) = val_test(
                    subtask, data['test'], self.model)
                logger.info(
                    f'Seed: {seed}, Subtask: {subtask}, ACC: {acc:.5f}, P: {P:.5f}, R: {R:.5f}, F1: {f1:.5f}, mP: {mP:.5f}')
                result[seed, subtask] = mP

        logger.info(result)
        logger.info(f'ave: {result.mean().item()}, {result.mean(dim=0)}')
        logger.info(f'End evaluation')
        return result.mean(dim=0)
