import os
import resource
from typing import Callable
from collections import defaultdict

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from einops import rearrange

def get_file_names(
    path: str,
    suffix: str | None = None,
    type: str | None = None,
    with_dir: bool = False,
    with_suffix: bool = False
) -> list[str]:
    """
    get file names by path, the file names will sorted, if file names are all number, they will sorted by numerically rather than lexicographically

    Example:
    dir structure:
        path:
            aaa.png
            bbb.png

    >>> print(get_path_file_names(path, '.png'))
    ['aaa', 'bbb']
    >>> print(get_path_file_names(path, '.png', with_dir=True))
    ['path/aaa', 'path/bbb']
    >>> print(get_path_file_names(path, '.png', with_suffix=True))
    ['aaa.png', 'bbb.png']

    Args:
        path: the path to get file names
        suffix: if suffix is not None: only file_name[-len(suffix):] == suffix will return, and return file name will not with suffix
        type: if type is dir: will only return dir name
        with_dir: all return file name will with its path
        with_suffix: all
    """

    file_names = os.listdir(path)
    all_num = True
    for file_name in file_names:
        if not file_name.split('.')[0].isdigit():
            all_num = False
            break

    if type == 'dir':
        file_names = list(filter(lambda file_name: os.path.isdir(os.path.join(path, file_name)), file_names))
    file_names = sorted(file_names, key=lambda x: int(x.split('.')[0])) if all_num else sorted(file_names)

    if suffix:
        file_names = [(file_name if with_suffix else file_name[:-len(suffix)]) for file_name in file_names if file_name[-len(suffix):] == suffix]

    if with_dir:
        file_names = [os.path.join(path, file_name) for file_name in file_names]

    return file_names


def mkdir(*args: str) -> None:
    for dir_name in args:
        if dir_name is None:
            continue
        if '.' in os.path.basename(dir_name):  # if is file
            dir_name = os.path.dirname(dir_name)
        if dir_name == '':
            continue
        os.makedirs(dir_name, exist_ok=True)


def read_avi(file_name: str):
    arr = []
    cap = cv2.VideoCapture(file_name)
    i = 1
    while cap.isOpened():
        _, frame = cap.read()
        if not _:
            break
        arr.append(frame)
        i += 1

    return np.stack(arr)


class NpyDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        df: pd.DataFrame | None = None,
        transform: Callable | None = None,
        rets: list[str] = ['image'],
        *,
        mmap_mode: str | None = None,
        cache: bool = False,
    ):
        super().__init__()

        self.dataset_path = dataset_path
        self.df = df
        self.transform = transform
        self.file_names = []
        if df is not None:
            file_names = df.index.tolist()
        else:
            file_names = get_file_names(f'{dataset_path}/{rets[0]}', '.npy')
            if len(file_names) == 0:
                file_names = get_file_names(f'{dataset_path}/{rets[0]}')
        for file_name in file_names:
            if os.path.isdir(f'{dataset_path}/{rets[0]}/{file_name}'):
                sub_file_names = get_file_names(f'{dataset_path}/{rets[0]}/{file_name}', '.npy')
                for sub_file_name in sub_file_names:
                    self.file_names.append(f'{file_name}/{sub_file_name}')
            elif os.path.exists(f'{dataset_path}/{rets[0]}/{file_name}.npy'):
                self.file_names.append(file_name)
            else:
                raise FileNotFoundError(f'Cant find {dataset_path}/{rets[0]}/{file_name} or {dataset_path}/{rets[0]}/{file_name}.npy')

        self.rets = rets

        self.mmap_mode = mmap_mode
        if mmap_mode == 'r':
            resource.setrlimit(resource.RLIMIT_NOFILE, (65536, 65536))
        self.cache = cache
        if self.cache:
            self.cache_dict = defaultdict(dict)
            for file_name in tqdm(self.file_names, desc='Dataset loading: '):
                for ret in self.rets:
                    if os.path.exists(f'{dataset_path}/{ret}/{file_name}.npy'):
                        self.cache_dict[file_name][ret] = np.load(f'{dataset_path}/{ret}/{file_name}.npy', mmap_mode=self.mmap_mode)

    def __len__(self):
        return len(self.file_names)

    def get_one_item(self, idx):
        file_name = self.file_names[idx]
        res = {}
        for ret in self.rets:
            if os.path.exists(f'{self.dataset_path}/{ret}/{file_name}.npy'):
                res[ret] = self.cache_dict[file_name][ret] if self.cache else np.load(f'{self.dataset_path}/{ret}/{file_name}.npy', mmap_mode=self.mmap_mode)
            elif ret == 'file_name':
                res['file_name'] = file_name
            elif file_name in self.df.index and ret in self.df:
                res[ret] = self.df.loc[file_name, ret]
            elif file_name.split('/')[0] in self.df.index and ret in self.df:
                res[ret] = self.df.loc[file_name.split('/')[0], ret]
            else:
                res[ret] = None

        return res

    def __getitem__(self, idx):
        res = self.get_one_item(idx)
        if self.transform:
            res = self.transform(res)

        return res


def replay_transform(transform, video):
    """
    Args:
        transform: Compose
        video: (T, H, W, 3), np.uint8

    Returns:
        video: (3, T, H, W), torch.float
    """
    x = []
    replay = None
    for i, image in enumerate(video):
        if i == 0:
            t = transform(image=image)
            replay = t['replay']
        else:
            t = transform.replay(replay, image=image)
        x.append(t['image'])

    video = torch.stack(x, dim=0).permute(1, 0, 2, 3)
    return video


def patchify(x, patch_size: int = 8):
    """
    Args:
        x: (B, c, T, H, W)

    Returns:
        x: (B, thw, p0p1p2c)
    """
    return rearrange(x, 'B c (t p0) (h p1) (w p2) -> B (t h w) (p0 p1 p2 c)', p0=2, p1=patch_size, p2=patch_size)


def get_roi():
    roi = ~np.array([
        [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1],
        [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype=bool)
    return roi


def random_mask_in_roi(roi: np.ndarray | None = None, mask_ratio: float = 0.75) -> np.ndarray:
    if roi is None:
        roi = get_roi().reshape(-1)

    num_token = roi.sum()
    num_unmask = int(num_token * (1 - mask_ratio))
    roi_indices = roi.nonzero()[0]
    mask = []
    roi_unmask_one_frame = np.concatenate([np.ones(num_unmask, dtype=bool), np.zeros(num_token - num_unmask, dtype=bool)], axis=0)
    for _ in range(8):
        roi_unmask_one_frame_cp = roi_unmask_one_frame.copy()
        np.random.shuffle(roi_unmask_one_frame_cp)
        roi_one_frame = roi.copy()
        roi_one_frame[roi_indices[roi_unmask_one_frame_cp]] = 0
        mask.append(roi_one_frame)
    mask = np.concatenate(mask, axis=0)

    return mask
