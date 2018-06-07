from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import os
import cv2
from glob import glob
import hgtk

char_list = (
    'ㄱ',
    'ㄴ',
    'ㄷ',
    'ㄹ',
    'ㅁ',
    'ㅂ',
    'ㅅ',
    'ㅇ',
    'ㅈ',
    'ㅊ',
    'ㅋ',
    'ㅌ',
    'ㅍ',
    'ㅎ',
    'ㅛ',
    'ㅕ',
    'ㅑ',
    'ㅐ',
    'ㅔ',
    'ㅗ',
    'ㅓ',
    'ㅏ',
    'ㅣ',
    'ㅠ',
    'ㅜ',
    'ㅡ',
    'ㅃ',
    'ㅉ',
    'ㄸ',
    'ㄲ',
    'ㅆ',
    'ㅒ',
    'ㅖ',
    '<sos>',
    '<eos>',
    '<pad>'
)

int_list = [i for i in range(len(char_list))]
one_hot = dict(zip(char_list, int_list))
to_char = dict(zip(int_list, char_list))

class videoDataset(Dataset):
    def __init__(self, path, videoMaxLen, txtMaxLen):
        self.queries = sorted(glob(os.path.join(path, '*/*/*.mp4')))
        self.labels = sorted(glob(os.path.join(path, '*/*/*.txt')))

        self.videoMaxLen = videoMaxLen
        self.txtMaxLen = txtMaxLen

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        return videoProcess(self.queries[idx], self.videoMaxLen), txtProcess(self.labels[idx], self.txtMaxLen)

def videoProcess(dir, videoMaxLen):
    cap = cv2.VideoCapture(dir)
    tmp = []
    results = torch.zeros(videoMaxLen, 120, 120)
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
            # Our operations on the frame come here
            gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (120, 120)).reshape(1, 120, 120)
            tmp.append(gray)
            # Display the resulting frame
            #imshow(gray, cmap='gray')
        else:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    results[:len(tmp), :, :] = torch.from_numpy(np.concatenate(tmp, axis=0).reshape(-1, 120, 120))
    cap.release()
    return results

def txtProcess(dir, txtMaxLen):
    result = []
    with open(dir) as f:
        tmp = [i for i in f.readline().replace(' ', '').replace('.', '').replace(',', '').replace('\"', '').replace('\'', '').rstrip('\n')]
        for i in tmp:
            result += [one_hot[i] for i in hgtk.letter.decompose(i)]
        result += [one_hot['<eos>']]
        if len(result) < txtMaxLen:
            result += [one_hot['<pad>'] for _ in range(txtMaxLen - len(tmp))]
        else:
            print(result)
            raise Exception('too short txt max length')
    # dataLen = len(result)
    # vector = torch.zeros((dataLen, txtMaxLen))
    
    # for i in range(dataLen):
    #     vector[i, np.arange(txtMaxLen), result[i]] = 1
    return torch.Tensor(tmp)