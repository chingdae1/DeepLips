import numpy as np
from glob import glob
import cv2
import sys
import os

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

def videoProcess(path):
    # dirs = glob(os.path.join(path, '*/*/*.mp4'))
    # largest = 0
    # for index, dir in enumerate(dirs):
    #     cap = cv2.VideoCapture(dir)
    #     tmp = []
    #     while(True):
    #         # Capture frame-by-frame
    #         ret, frame = cap.read()
    #         if ret:
    #             # Our operations on the frame come here
    #             gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (120, 120)).reshape(1, 120, 120)
    #             tmp.append(gray)
    #             # Display the resulting frame
    #             #imshow(gray, cmap='gray')
    #         else:
    #             break
    #         if cv2.waitKey(1) & 0xFF == ord('q'):
    #             break
    #     if largest < len(tmp):
    #         largest = len(tmp)
    # cap.release()
    # print('longest video', largest)

    longest = 0
    dirr = ''
    dirs = glob(os.path.join(path, '*/*/*.txt'))
    dirs = sorted(dirs)
    for dir in dirs:
        with open(dir) as f:
            tmp = [one_hot[i] for i in f.readline().split(':')[1].replace(' ', '').rstrip('\n')] + [one_hot['<eos>']]
            
            if longest < len(tmp):
                longest = len(tmp)
                dirr = dir
    print('longest text', longest)
    print(dirr)

if __name__ == '__main__':
    path = sys.argv[1]
    videoProcess(path)