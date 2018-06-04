import cv2
import numpy as np
import os
from glob import glob, iglob
import sys
import dlib
from collections import deque
import shutil

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')

### set dir path
# main_path = './main/'
# pretrain_path = './pretrain/'

#result_path = './result/'

cv2.namedWindow('show', 0)

def crop_lip(video_path, result_path):
    main_path
    vc = cv2.VideoCapture(video_path)
    # print (vc.get(cv2.CAP_PROP_FPS)) # all video have 25 fps
    root_name = video_path.split('/')[-3] # main / pretrain 
    dir_name = video_path.split('/')[-2]
    file_name = os.path.basename(video_path)
    print (dir_name, file_name, sep=',', end=',')

    save_path = os.path.join(result_path, root_name, dir_name)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    fourcc = cv2.VideoWriter_fourcc(*'x264')
    vw = cv2.VideoWriter(os.path.join(save_path, file_name), fourcc, 25.0, (120,120))

    # track bbox
    queue = deque(maxlen=5)
    while True:
        img_bgr = vc.read()[1]
        if img_bgr is None:
            break
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        dlib_rects = detector(img_rgb, 1) # dlib bbox detection

        ### face must be found as one.
        if len(dlib_rects) == 1:
            dlib_rect = max(dlib_rects, key=lambda rect: rect.width() * rect.height())
            points = predictor(img_rgb, dlib_rect)
            landmarks = list(map(lambda p: (p.x, p.y), points.parts()))
            
            x,y,w,h = (dlib_rect.left(), dlib_rect.top(), dlib_rect.width(), dlib_rect.height())
            bbox_xywh = (x,y,w,h)
            queue.append(bbox_xywh)
            flg_tracking = False

        ### when face is not found & not tracked
        elif len(queue) < 1:
            dlib_rect = None
            landmarks = []
            flg_tracking = False
            print ('cannot find face')#, file=sys.stderr)
            vw.release()
            if os.path.exists(os.path.join(save_path, file_name)):
                os.remove(os.path.join(save_path, file_name))
            return

        ### not detected but tracked
        else:
            bbox_xywh = queue[-1] # prev bbox(curr) -->  prev landmark box(TBC)
            x,y,w,h = bbox_xywh
            dlib_rect = dlib.rectangle(left=x,top=y,right=x+w,bottom=y+h)
            points = predictor(img_rgb, dlib_rect)
            landmarks = list(map(lambda p: (p.x, p.y), points.parts()))
        
        ### lip part: lower lip center is the standard point(66 point)
        margin = 30
        roiX1 = landmarks[66][0] - margin if landmarks[66][0] - margin > 0 else 0
        roiY1 = landmarks[66][1] - margin if landmarks[66][1] - margin > 0 else 0
        roiX2 = landmarks[66][0] + margin if landmarks[66][0] + margin < img_bgr.shape[1] else img_bgr.shape[1]
        roiY2 = landmarks[66][1] + margin if landmarks[66][1] + margin < img_bgr.shape[0] else img_bgr.shape[0]
        img_lip = img_bgr[roiY1:roiY2, roiX1:roiX2].copy()
        img_lip = cv2.resize(img_lip, (120,120))
        
        if roiX2-roiX1 != roiY2-roiY1:
            print ((roiX1, roiY1, roiX2-roiX1, roiY2-roiY1))
            return 
        vw.write(img_lip)

        ### draw image
        if flg_tracking == True:
            bbox_color = (0,0,255)
        else:
            bbox_color = (0,255,0)
        
        x,y,w,h = bbox_xywh
        cv2.rectangle(img_bgr, (x,y), (x+w,y+h), bbox_color, 1) # face box
        
        for mark in landmarks:
            cv2.circle(img_bgr, (mark[0], mark[1]), 1, (0,0,255), -1) # marks
        
        cv2.rectangle(img_bgr, (roiX1,roiY1), (roiX2,roiY2), (255,255,255), 2) # lip box
        # print ((roiX1, roiY1, roiX2-roiX1, roiY2-roiY1))
        
        cv2.imshow('show', img_bgr)
        cv2.imshow('lip', img_lip)
        
        key = cv2.waitKey(1)
        if key == 27:
            exit()
    print('done')
    vw.release()
    

def main(result_path, data_path):
    ### mp4 path list
    list_video_in_main = sorted(glob(data_path+'/main/'+'*/*.mp4'))
    list_video_in_pretrain = sorted(glob(data_path+'/pretrain/'+'*/*.mp4'))
    
    list_txt_in_main = glob(data_path+'/main/'+'*/*.txt')
    list_txt_in_pretrain = glob(data_path+'/pretrain/'+'*/*.txt')
    
    for path in list_video_in_pretrain:
        crop_lip(path, result_path)

    for path in list_video_in_main:
        crop_lip(path, result_path)

    result_video = glob(result_path + '*/*/*.mp4')

    for path in list_txt_in_main:
        root_name = path.split('/')[-3] # main / pretrain 
        dir_name = path.split('/')[-2]
        file_name = os.path.basename(path)
        save_path = os.path.join(result_path, root_name, dir_name)
        if os.path.join(save_path, file_name).replace('.txt', '.mp4') in result_video:
            shutil.move(path, save_path)
        

    for path in list_txt_in_pretrain:
        root_name = path.split('/')[-3] # main / pretrain 
        dir_name = path.split('/')[-2]
        file_name = os.path.basename(path)
        save_path = os.path.join(result_path, root_name, dir_name)
        if os.path.join(save_path, file_name).replace('.txt', '.mp4') in result_video:
            shutil.move(path, save_path)
   
if __name__ == '__main__':
    data_path = sys.argv[1]
    #/mnt/disks/new_disk
    result_path = sys.argv[2]
    #/home/chingdae123/DeepLips/data/data_cut
    main(result_path, data_path)