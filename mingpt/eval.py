import copy
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from utils import get_time_str


class EvalDataset(Dataset):
    """Evaluation dataset.
    """

    def __init__(self, feats, labels):
        super(EvalDataset, self).__init__()
        self.feats = feats
        self.labels = labels

    def __getitem__(self, i):
        return {
            'feat': self.feats,
            'label': self.labels,
        }

    def __len__(self):
        return 1


class Head(nn.Module):
    def __init__(self, config):
        super().__init__()
        input_dim = config['input_dim']
        output_dim = config['output_dim']
        self.fc = nn.Linear(input_dim, output_dim, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x):
        return self.fc(x)


def create_dataloader(feats, labels, indexes, config):
    feats = feats[indexes]
    labels = labels[indexes]
    dataset = EvalDataset(feats, labels)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False)
    return dataloader


def train(subtask, feats, labels, train_indexes, val_indexes, init_state_dict, lr, config):
    # create dataloader
    train_dataloader = create_dataloader(feats, labels, train_indexes, config)
    val_dataloader = create_dataloader(feats, labels, val_indexes, config)
    # create model
    model = Head(config)
    model.load_state_dict(init_state_dict)
    optimizer = optim.SGD(model.parameters(), lr, momentum=0.9)
    device = torch.cuda.current_device()
    model = nn.DataParallel(model).to(device)
    best_metric = 0
    best_state_dict = init_state_dict
    for e in range(20):
        model.train()
        for data in train_dataloader:
            feat = data['feat'].squeeze(0)
            label = data['label'].squeeze(0)[:, subtask]
            logit = model(feat)
            loss = F.cross_entropy(logit, label)
            model.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        logits = []
        labels = []
        with torch.set_grad_enabled(False):
            for data in val_dataloader:
                feat = data['feat'].squeeze(0)
                label = data['label'].squeeze(0)[:, subtask]
                logit = model(feat)
                logits.append(logit)
                labels.append(label)
            logits = torch.cat(logits, dim=0).cpu()
            labels = torch.cat(labels, dim=0).cpu()
            (acc, P, R, f1) = cal_metric(logits, labels)
            # print(f'epoch: {e + 1}, acc: {acc:.5f}, P: {P:.5f}, R: {R:.5f}, f1: {f1:.5f}')
            if f1 > best_metric:
                best_metric = f1
                best_state_dict = copy.deepcopy(model.module.state_dict())
    return (best_metric, best_state_dict)


@torch.no_grad()
def test(subtask, feats, labels, test_indexes, state_dict, config):
    # create dataloader
    dataloader = create_dataloader(feats, labels, test_indexes, config)
    # create model
    model = Head(config)
    model.load_state_dict(state_dict)
    device = torch.cuda.current_device()
    model = nn.DataParallel(model).to(device)
    model.eval()
    logits = []
    labels = []
    for data in dataloader:
        feat = data['feat'].squeeze(0)
        label = data['label'].squeeze(0)[:, subtask]
        logit = model(feat)
        logits.append(logit)
        labels.append(label)
    logits = torch.cat(logits, dim=0).cpu()
    labels = torch.cat(labels, dim=0).cpu()
    return (logits, labels)


def cal_metric(logits, labels):
    preds = torch.argmax(logits, dim=1)
    TP, FN, FP, TN = 0, 0, 0, 0
    TP = ((labels == 1) & (preds == 1)).sum().item()
    FN = ((labels == 1) & (preds == 0)).sum().item()
    FP = ((labels == 0) & (preds == 1)).sum().item()
    TN = ((labels == 0) & (preds == 0)).sum().item()
    acc, P, R, f1 = 0, 0, 0, 0
    if (TP + FN + FP + TN) != 0:
        acc = TP / (TP + FN + FP + TN)
    if (TP + FP) != 0:
        P = TP / (TP + FP)
    if (TP + TN) != 0:
        R = TP / (TP + TN)
    if (P + R) != 0:
        f1 = 2 * P * R / (P + R)
    return (acc, P, R, f1)


class Evaluator:
    def __init__(self, config):
        self.config = config
        self.num_seeds = config['num_seeds']
        self.num_subtasks = config['num_subtasks']
        self.lr_list = config['lr_list']
        # data
        self.num_samples = config['num_samples']
        self.indexes_list = []
        for i in range(self.num_seeds):
            indexes = list(range(self.num_samples))
            random.shuffle(indexes)
            self.indexes_list.append(indexes)
        self.model = Head(config)
        self.init_state_dict = copy.deepcopy(self.model.state_dict())

    def eval(self, feats, labels):
        """num_seeds * num_subtasks * num_lrs = 3 * 2 * 9
        """
        print(f'{get_time_str()}, start evaluation')
        result = torch.zeros((self.num_seeds, self.num_subtasks))
        # 1. loop seeds
        for i, indexes in enumerate(self.indexes_list):
            print(f'{get_time_str()}, seed: {i}')
            # 2. split train, val and test
            num = self.num_samples
            num_train = int(num * 0.6)
            num_val = int(num * 0.2)
            train_indexes = indexes[: num_train]
            val_indexes = indexes[num_train: num_train + num_val]
            test_indexes = indexes[num_train + num_val:]
            # 3. loop subtasks
            for j in range(self.num_subtasks):
                print(f'{get_time_str()}, subtask: {j}')
                # 4. loop lr
                best_metric = 0.
                best_state_dict = self.init_state_dict
                for lr in self.lr_list:
                    # 5. train subtask model
                    (metric, state_dict) = train(j, feats, labels, train_indexes, val_indexes, self.init_state_dict, lr, self.config)
                    if metric > best_metric:
                        best_metric = metric
                        best_state_dict = copy.deepcopy(state_dict)
                    print(f'lr: {lr:e}, metric: {metric:.5f}, best_metric: {best_metric:.5f}')
                # 6. test
                (acc, P, R, f1) = cal_metric(*test(j, feats, labels, test_indexes, best_state_dict, self.config))
                result[i, j] = f1
                print(f'{get_time_str()}, seed_{i}, subtask_{j}, acc: {acc:.5f}, P: {P:.5f}, R: {R:.5f}, f1: {f1:.5f}')
        print(result)
        print(f'{get_time_str()}, ave:', result.mean())
        print(f'{get_time_str()}, end evaluation')
        return result.mean().item()
