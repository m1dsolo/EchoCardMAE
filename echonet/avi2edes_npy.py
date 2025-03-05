import os
import skimage
import numpy as np
import pandas as pd

from utils import mkdir

trace_df = pd.read_csv(f'/home/yang/dataset/EchoNet-Dynamic/VolumeTracings.csv')
trace_df['FileName'] = trace_df['FileName'].str.replace(r'\.\w+$', '', regex=True)
trace_df = trace_df.groupby('FileName')

OUT_PATH = f'./npy/echonet/edes'

file_names = list(trace_df.groups.keys())
for i, file_name in enumerate(file_names, 1):
    npy_name = f'./npy/echonet/main/video/{file_name}.npy'
    if not os.path.exists(npy_name):
        print(f'{i}/{len(file_names)}, {file_name} not found')
        continue
    video = np.load(npy_name) # (D, H, W, 3)

    edes_df = trace_df.get_group(file_name).groupby('Frame')
    assert len(edes_df) == 2

    for j, (frame, df) in enumerate(edes_df):
        image = video[frame]

        x1 = df['X1'].values
        y1 = df['Y1'].values
        x2 = df['X2'].values
        y2 = df['Y2'].values

        x = np.concatenate((x1[1:], np.flip(x2[1:])))
        y = np.concatenate((y1[1:], np.flip(y2[1:])))

        mask = np.zeros(video.shape[1:3], bool)
        r, c = skimage.draw.polygon(np.rint(y), np.rint(x), mask.shape)
        mask[r, c] = 1

        name = ['ed', 'es'][j]
        mkdir(f'{OUT_PATH}/image/{file_name}')
        mkdir(f'{OUT_PATH}/mask/{file_name}')
        np.save(f'{OUT_PATH}/image/{file_name}/{name}.npy', image)
        np.save(f'{OUT_PATH}/mask/{file_name}/{name}.npy', mask)

    print(f'{i}/{len(file_names)}, {file_name}')
