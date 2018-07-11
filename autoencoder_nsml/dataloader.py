from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch
import numpy as np
import os
import cv2
from glob import glob


class VideoDataSet(Dataset):
    def __init__(self, data, videomaxlen):
        self.queries = data
        self.videoMaxLen = videomaxlen

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        return video_process(self.queries[idx], self.videoMaxLen)


def video_process(dir, videomaxlen):
    cap = cv2.VideoCapture(dir)
    tmp = []
    results = torch.zeros(videomaxlen, 120, 120)
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
            # Our operations on the frame come here
            gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (120, 120)).reshape(1, 120, 120)
            tmp.append(gray)
            # Display the resulting frame
            # imshow(gray, cmap='gray')
        else:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    try:
        results[:len(tmp), :, :] = torch.from_numpy(np.concatenate(tmp, axis=0))
    except:
        print('#############################', dir)

    cap.release()

    return results


def get_data_loader(path, bs, validationratio, videomaxlen):
    files = sorted(glob(os.path.join(path, '*.mp4')))
    start = int(len(files) * validationratio)

    train_dataset = VideoDataSet(files[:-start], videomaxlen)
    val_dataset = VideoDataSet(files[-start:], videomaxlen)

    train_loader = DataLoader(
        train_dataset, batch_size=bs, shuffle=True, pin_memory=True, num_workers=3)

    val_loader = DataLoader(
        val_dataset, batch_size=bs, shuffle=True, pin_memory=True, num_workers=8)

    return train_loader, val_loader
