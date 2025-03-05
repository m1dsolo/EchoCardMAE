import numpy as np

from utils import (
    get_file_names,
    mkdir,
    read_avi
)

AVI_PATH = 'dataset/EchoNet-Dynamic/a4c-video-dir'
OUT_PATH = 'npy/echonet/main/video'
mkdir(OUT_PATH)

file_names = get_file_names(AVI_PATH, '.avi')
for i, file_name in enumerate(file_names, 1):
    print(f'{i}/{len(file_names)}:')

    images = read_avi(f'{AVI_PATH}/{file_name}.avi') # (D, H, W, 3)
    print(f'{file_name}, {images.shape}')

    np.save(f'{OUT_PATH}/{file_name}.npy', images)
