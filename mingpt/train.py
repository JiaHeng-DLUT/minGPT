import argparse
import datetime
import logging
import math
import time
import torch
from os import path as osp

from mingpt.data import create_dataloader, create_dataset
from mingpt.models import create_model
from mingpt.utils import (MessageLogger, check_resume, get_env_info,
                          get_root_logger, get_time_str, init_tb_logger,
                          init_wandb_logger, make_exp_dirs, mkdir_and_rename,
                          set_random_seed)
from mingpt.utils.dist_util import get_dist_info, init_dist
from mingpt.utils.options import dict2str, parse


def parse_options(is_train=True):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-opt', type=str, required=True, help='Path to option YAML file.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    opt = parse(args.opt, is_train=is_train)

    # distributed settings
    if args.launcher == 'none':
        opt['dist'] = False
        print('Disable distributed.', flush=True)
    else:
        opt['dist'] = True
        if args.launcher == 'slurm' and 'dist_params' in opt:
            init_dist(args.launcher, **opt['dist_params'])
        else:
            init_dist(args.launcher)

    opt['rank'], opt['world_size'] = get_dist_info()

    # random seed
    seed = opt.get('manual_seed')
    assert seed is not None, 'Seed must be set.'
    set_random_seed(seed + opt['rank'])

    return opt


def init_loggers(opt):
    log_file = osp.join(opt['path']['log'],
                        f"train_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(
        logger_name='mingpt', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    # initialize tensorboard logger and wandb logger
    tb_logger = None
    if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name']:
        tb_logger = init_tb_logger(log_dir=osp.join('tb_logger', opt['name']))
    if (opt['logger'].get('wandb')
            is not None) and (opt['logger']['wandb'].get('project')
                              is not None) and ('debug' not in opt['name']):
        assert opt['logger'].get('use_tb_logger') is True, (
            'should turn on tensorboard when using wandb')
        init_wandb_logger(opt)
    return logger, tb_logger


def create_train_val_dataloader(opt, logger):
    # create train and val dataloaders
    train_loader, val_loader = None, None
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = create_dataset(dataset_opt)
            train_loader = create_dataloader(
                train_set,
                dataset_opt,
                seed=opt['manual_seed'],
                phase=phase)

            num_iter_per_epoch = math.ceil(
                len(train_set) / (dataset_opt['batch_size_per_gpu'] * opt['world_size']))
            total_iters = int(opt['train']['total_iter'])
            total_epochs = math.ceil(total_iters / (num_iter_per_epoch))
            logger.info(
                'Training statistics:'
                f'\n\tNumber of train samples: {len(train_set)}'
                f'\n\tBatch size per gpu: {dataset_opt["batch_size_per_gpu"]}'
                f'\n\tWorld size (gpu number): {opt["world_size"]}'
                f'\n\tRequire iter number per epoch: {num_iter_per_epoch}'
                f'\n\tTotal epochs: {total_epochs}; iters: {total_iters}.')

        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(
                val_set,
                dataset_opt,
                seed=opt['manual_seed'],
                phase=phase)
            logger.info(
                f'Number of val samples in {dataset_opt["name"]}: '
                f'{len(val_set)}')
        else:
            raise ValueError(f'Dataset phase {phase} is not recognized.')

    return train_loader, val_loader, total_epochs, total_iters


def main():
    # parse options, set distributed setting, set ramdom seed
    opt = parse_options(is_train=True)

    # torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # load resume states if necessary
    if opt['path'].get('resume_state'):
        device_id = torch.cuda.current_device()
        resume_state = torch.load(
            opt['path']['resume_state'],
            map_location=lambda storage, loc: storage.cuda(device_id))
    else:
        resume_state = None

    # mkdir for experiments and logger
    if resume_state is None:
        make_exp_dirs(opt)
        if opt['logger'].get('use_tb_logger') and 'debug' not in opt[
                'name'] and opt['rank'] == 0:
            mkdir_and_rename(osp.join('tb_logger', opt['name']))

    # initialize loggers
    logger, tb_logger = init_loggers(opt)

    # create train and validation dataloaders
    result = create_train_val_dataloader(opt, logger)
    train_loader, val_loader, total_epochs, total_iters = result

    # create model
    if resume_state:  # resume training
        check_resume(opt, resume_state['iter'])
        model = create_model(opt)
        model.resume_training(resume_state)  # handle optimizers and schedulers
        logger.info(f"Resuming training from epoch: {resume_state['epoch']}, "
                    f"iter: {resume_state['iter']}.")
        start_epoch = resume_state['epoch']
        current_iter = resume_state['iter']
    else:
        model = create_model(opt)
        start_epoch = 0
        current_iter = 0

    # create message logger (formatted outputs)
    msg_logger = MessageLogger(opt, current_iter, tb_logger)

    # training
    logger.info(
        f'Start training from epoch: {start_epoch}, iter: {current_iter}')
    data_time, iter_time = time.time(), time.time()
    start_time = time.time()

    model.validation(val_loader, current_iter, tb_logger)
    for epoch in range(start_epoch, total_epochs + 1):
        for train_data in train_loader:
            data_time = time.time() - data_time

            current_iter += 1
            if current_iter > total_iters:
                break
            # # update learning rate
            # model.update_learning_rate(
            #     current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))
            # # training
            # model.feed_data(train_data)
            # model.optimize_parameters(current_iter)
            # iter_time = time.time() - iter_time
            # # log
            # if current_iter % opt['logger']['print_freq'] == 0:
            #     log_vars = {'epoch': epoch, 'iter': current_iter}
            #     log_vars.update({'lrs': model.get_current_learning_rate()})
            #     log_vars.update({'time': iter_time, 'data_time': data_time})
            #     log_vars.update(model.get_current_log())
            #     msg_logger(log_vars)

            # # save models and training states
            # if current_iter % opt['logger']['save_checkpoint_freq'] == 0:
            #     logger.info('Saving models and training states.')
            #     model.save(epoch, current_iter)
            if current_iter % opt['logger']['save_checkpoint_freq'] == 0:
                model.load_network(model.net, f'experiments/m20_fix_eval_bug_base04/models/net_{current_iter}.pth')

            # validation
            if opt.get('val') is not None and (current_iter %
                                               opt['val']['val_freq'] == 0):
                model.validation(val_loader, current_iter, tb_logger)

            data_time = time.time()
            iter_time = time.time()
        # end of iter

    # end of epoch

    consumed_time = str(
        datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f'End of training. Time consumed: {consumed_time}')
    logger.info('Save the latest model.')
    model.save(epoch=-1, current_iter=-1)  # -1 stands for the latest
    if opt.get('val') is not None:
        model.validation(val_loader, current_iter, tb_logger)
    if tb_logger:
        tb_logger.close()


if __name__ == '__main__':
    main()
