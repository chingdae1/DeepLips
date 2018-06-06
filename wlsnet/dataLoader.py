from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import os
import cv2
from glob import glob

char_list = (
    'A',
    'B',
    'C',
    'D',
    'E',
    'F',
    'G',
    'H',
    'I',
    'J',
    'K',
    'L',
    'M',
    'N',
    'O',
    'P',
    'Q',
    'R',
    'S',
    'T',
    'U',
    'V',
    'W',
    'X',
    'Y',
    'Z',
    '1',
    '2',
    '3',
    '4',
    '5',
    '6',
    '7',
    '8',
    '9',
    '0',
    '<sos>',
    '<eos>',
    '<pad>',
    '\'',
)

int_list = [i for i in range(len(char_list))]
one_hot = dict(zip(char_list, int_list))
to_int = dict(zip(int_list, char_list))

class videoDataset(Dataset):
    def __init__(self, path, videoMaxLen, txtMaxLen):
        self.queries = videoProcess(path, videoMaxLen)
        self.labels = txtProcess(path, txtMaxLen)

    def __len__(self):
        return self.queries.size()[0]

    def __getitem__(self, idx):
        return self.queries[idx], self.labels[idx]

def videoProcess(path, videoMaxLen):
    dirs = glob(os.path.join(path, '*/*.mp4'))
    dirs = sorted(dirs)
    results = torch.zeros(len(dirs), videoMaxLen, 120, 120)
    for index, dir in enumerate(dirs):
        cap = cv2.VideoCapture(dir)
        tmp = []
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
        results[index, :len(tmp), :, :] = torch.from_numpy(np.concatenate(tmp, axis=0).reshape(1, -1, 120, 120))
    cap.release()
    return results

def txtProcess(path, txtMaxLen):
    result = []
    dirs = glob(os.path.join(path, '*/*.txt'))
    dirs = sorted(dirs)
    for dir in dirs:
        with open(dir) as f:
            tmp = [one_hot[i] for i in f.readline().split(':')[1].replace(' ', '').rstrip('\n')] + [one_hot['<eos>']]
            
            if len(tmp) < txtMaxLen:
                result.append(tmp+[one_hot['<pad>'] for _ in range(txtMaxLen - len(tmp))])
            
            else:
                print(tmp)
                raise Exception('too short txt max length')
    # dataLen = len(result)
    # vector = torch.zeros((dataLen, txtMaxLen))
    
    # for i in range(dataLen):
    #     vector[i, np.arange(txtMaxLen), result[i]] = 1
    return torch.Tensor(result)

# def to_int():
#     data_dict = dict(zip(int_list, char_list))

#     return data_dict[char]