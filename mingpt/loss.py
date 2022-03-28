import torch.nn.functional as F


def cal_animal_loss(pred, target):
    '''
    pred:   (b, t, c, d)
    target: (b, t, c, d)
    '''
    return F.mse_loss(pred, target)


def cal_frame_loss(pred, target):
    '''
    pred:   (b, t, d)
    target: (b, t, d)
    '''
    return F.mse_loss(pred, target)
