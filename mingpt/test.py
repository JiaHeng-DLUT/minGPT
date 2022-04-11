import logging
import numpy as np
import torch
from os import path as osp
from tqdm import tqdm

from mingpt.data import create_dataloader, create_dataset
from mingpt.models import create_model
from mingpt.train import parse_options
from mingpt.utils import (get_env_info, get_root_logger, get_time_str,
                          make_exp_dirs)
from mingpt.utils.options import dict2str


def validate_submission(submission, submission_clips):
    if not isinstance(submission, dict):
        print("Submission should be dict")
        return False

    if 'frame_number_map' not in submission:
        print("Frame number map missing")
        return False

    if 'embeddings' not in submission:
        print('Embeddings array missing')
        return False
    elif not isinstance(submission['embeddings'], np.ndarray):
        print("Embeddings should be a numpy array")
        return False
    elif not len(submission['embeddings'].shape) == 2:
        print("Embeddings should be 2D array")
        return False
    elif not submission['embeddings'].shape[1] <= 256:
        print("Embeddings too large, max allowed is 256")
        return False
    elif not isinstance(submission['embeddings'][0, 0], np.float32):
        print(f"Embeddings are not float32")
        return False

    total_clip_length = 0
    for key in submission_clips['sequences']:
        start, end = submission['frame_number_map'][key]
        clip_length = submission_clips['sequences'][key]['keypoints'].shape[0]
        total_clip_length += clip_length
        if not end-start == clip_length:
            print(f"Frame number map for clip {key} doesn't match clip length")
            return False

    if not len(submission['embeddings']) == total_clip_length:
        print(f"Emebddings length doesn't match submission clips total length")
        return False

    if not np.isfinite(submission['embeddings']).all():
        print(f"Emebddings contains NaN or infinity")
        return False

    print("All checks passed")
    return True


def main():
    # parse options, set distributed setting, set ramdom seed
    opt = parse_options(is_train=False)

    # torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # mkdir and initialize loggers
    make_exp_dirs(opt)
    log_file = osp.join(opt['path']['log'],
                        f"test_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(
        logger_name='mingpt', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    # create model
    model = create_model(opt)

    # create test dataset and dataloader
    for phase, dataset_opt in sorted(opt['datasets'].items()):
        test_set = create_dataset(dataset_opt)
        test_loader = create_dataloader(
            test_set,
            dataset_opt,
            seed=opt['manual_seed'],
            phase=phase)
        logger.info(
            f"Number of test samples in {dataset_opt['name']}: {len(test_set)}")

        test_set_name = dataset_opt['name']
        logger.info(f'Testing {test_set_name}...')
        feats = []
        for val_data in tqdm(test_loader):
            model.feed_data(val_data)
            model.test()
            feat = model.output
            feat = feat.view(-1, feat.shape[-1])
            feats.append(feat)
        feats = torch.cat(feats, dim=0).cpu().numpy()
        print(feats.shape)

        frame_number_map = np.load(
            dataset_opt['frame_number_map_path'], allow_pickle=True).item()
        submission_dict = {
            "frame_number_map": frame_number_map,
            "embeddings": feats
        }
        submission_clips = np.load(
            dataset_opt['data_path'], allow_pickle=True).item()
        validate_submission(submission_dict, submission_clips)
        np.save(opt['path']['pretrain_network'] +
                '_submission_wo_mask.npy', submission_dict)


if __name__ == '__main__':
    main()
