from __future__ import print_function
from glob import glob
import os
import cv2
import argparse
from preprocessor import face_crop

parser = argparse.ArgumentParser()
parser.add_argument('--with_draw', help='do draw?', default='False')
parser.add_argument('--result_path', help='result path', default='./result')
parser.add_argument('--data_path', help='data path', default='./data')

args = parser.parse_args()

fourcc = cv2.VideoWriter_fourcc(*'avc1')

list_file_path = sorted(glob(os.path.join(args.data_path, '*.mp4')))

for file_path in list_file_path:
    basename = os.path.basename(file_path)
    vc = cv2.VideoCapture(file_path)
    # vw_face = cv2.VideoWriter()
    vw_face = cv2.VideoWriter(
                os.path.join(args.result_path,'face',basename), 
                fourcc, 30.0, (64,64))
    vw_lip = cv2.VideoWriter(
                os.path.join(args.result_path,'lip',basename), 
                fourcc, 30.0, (120,120))
    
    face_crop(vc, args, vw_face, vw_lip)
    
    vw_face.close()
    vw_lip.close()
