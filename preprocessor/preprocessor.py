from __future__ import print_function
import numpy as np
import cv2
import sys
import dlib
import argparse

net = cv2.dnn.readNetFromCaffe('./models/deploy.prototxt.txt', './models/res10_300x300_ssd_iter_140000.caffemodel')
detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor('./models/shape_predictor_68_face_landmarks.dat')

img_black_lip = np.zeros((120,120,3), dtype=np.uint8)
img_black_face = np.zeros((64,64,3), dtype=np.uint8)

### ocv dnn module preprocessor: gamma correction
def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
 
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

def face_crop(vc, args, vw_face=None, vw_lip=None):
    flg_multiface_detected = False
    idx = 0
    while True:
        idx += 1
        bgr_img = vc.read()[1]
        if bgr_img is None:
            break
        # if idx%2 != 0: continue
        # if idx < 30 * 12: continue
        bgr_img_origin = bgr_img.copy()
        bgr_img = cv2.resize(bgr_img, None, fx=.5, fy=.5)

        ### analysis
        bgr_img_proc = bgr_img.copy()
        gray_img = cv2.cvtColor(bgr_img_proc, cv2.COLOR_BGR2GRAY)
        # print ('before: %.3f'%gray_img.mean(), end=', ')
        for i in range(3):
            gray_img = cv2.cvtColor(bgr_img_proc, cv2.COLOR_BGR2GRAY)
            # print (i, end=', ')
            if gray_img.mean() < 130:
                bgr_img_proc = adjust_gamma(bgr_img_proc, 1.5)
            else:
                break

        ### ocv detection
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        start = cv2.getTickCount()
        (h, w) = bgr_img.shape[:2]
        
        blob = cv2.dnn.blobFromImage(cv2.resize(bgr_img_proc, (300, 300)), 
                                    1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward() # detect many roi

        ### ocvdnn --> bbox
        list_bboxes = []
        list_confidence = []
        list_dlib_rect = []
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence < 0.65:
                    continue
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (l, t, r, b) = box.astype("int") # l t r b
            
            original_vertical_length = b-t
            t = int(t + (original_vertical_length)*0.15)
            b = int(b - (original_vertical_length)*0.05)

            margin = ((b-t) - (r-l))//2
            l = l - margin if (b-t-r+l)%2 == 0 else l - margin - 1
            r = r + margin
            list_bboxes.append([l, t, r, b])
            list_confidence.append(confidence)
            rect_bb = dlib.rectangle(left=l, top=t, right=r, bottom=b)
            list_dlib_rect.append(rect_bb)

        # if box size is 0, use dlib detector
        # if box size > 1, make flg_multiface TRUE
        # and choose max size bbox
        if len(list_bboxes) == 0:
            dlib_rects = detector(rgb_img, 1) # dlib bbox detection
            if len(dlib_rects) > 1: flg_multiface_detected = True
            elif len(dlib_rects) == 1: flg_multiface_detected = False
            if len(dlib_rects) != 0:
                dlib_rect = max(dlib_rects, key=lambda rect: rect.width() * rect.height())
                
                list_dlib_rect = [dlib_rect]
                list_confidence = [1.]
                list_bboxes = [[dlib_rect.left(), dlib_rect.top(), dlib_rect.right(), dlib_rect.bottom()]]
        
        elif len(list_bboxes) > 1:
            zip_box = zip(list_bboxes, list_confidence, list_dlib_rect)
            max_box = max(zip_box, key=lambda rect: rect[0][2] - rect[0][0])
            
            list_bboxes = [max_box[0]]
            list_confidence = [max_box[1]]
            list_dlib_rect = [max_box[2]]
            flg_multiface_detected = True
        
        elif len(list_bboxes) == 1:
            flg_multiface_detected = False

        ### landmark
        list_landmarks = []
        for rect in list_dlib_rect:
            points = landmark_predictor(rgb_img, rect)
            list_points = list(map(lambda p: (p.x, p.y), points.parts()))
            list_landmarks.append(list_points)

        time = (cv2.getTickCount() - start) / cv2.getTickFrequency() * 1000

        ### lip cropping
        if len(list_landmarks) != 0:
            for bboxes, landmarks in zip(list_bboxes, list_landmarks):
                ### lip part: lower lip center is the standard point(66 point)
                margin = int(round((bboxes[2]-bboxes[0]) / 3.5))
                roiX1 = landmarks[66][0] - margin if landmarks[66][0] - margin > 0 else 0
                roiY1 = landmarks[66][1] - margin if landmarks[66][1] - margin > 0 else 0
                roiX2 = landmarks[66][0] + margin if landmarks[66][0] + margin < bgr_img.shape[1] else bgr_img.shape[1]
                roiY2 = landmarks[66][1] + margin if landmarks[66][1] + margin < bgr_img.shape[0] else bgr_img.shape[0]
                # img_lip = bgr_img[roiY1:roiY2, roiX1:roiX2].copy()
                img_face = bgr_img_origin[t*2:b*2, l*2:r*2].copy()
                img_lip = bgr_img_origin[roiY1*2:roiY2*2, roiX1*2:roiX2*2].copy()
                img_face = cv2.resize(img_face, (64,64))
                img_lip = cv2.resize(img_lip, (120,120))
                l,t,r,b = bboxes
                
                if vw_face is not None:
                    vw_face.write(img_face)
                if vw_lip is not None:
                    if flg_multiface_detected == True:
                        vw_lip[-1][-1] = [0,0,255]
                    vw_lip.write(img_lip)
                
                if roiX2-roiX1 != roiY2-roiY1:
                    print ((roiX1, roiY1, roiX2-roiX1, roiY2-roiY1), file=sys.stderr)
        else:
            if vw_face is not None:
                vw_face.write(img_black_face)
            if vw_lip is not None:
                vw_lip.write(img_black_lip)

        ### analysis ocv dnn brightness
        gray_img = cv2.cvtColor(bgr_img_proc, cv2.COLOR_BGR2GRAY)
        # print ('after: %.3f'%gray_img.mean(), end=', ')

        print ('%d, elapsed time: %.3fms'%(idx,time))

        ### draw rectangle bbox
        if args.with_draw == 'True':
            for bbox, confidence in zip(list_bboxes, list_confidence):
                l, t, r, b = bbox
                
                cv2.rectangle(bgr_img, (l, t), (r, b),
                    (255, 255, 0), 2)
                text = "face: %.2f" % confidence
                text_size, base_line = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                y = t #- 1 if t - 1 > 1 else t + 1
                cv2.rectangle(bgr_img, (l,y-text_size[1]),(l+text_size[0], y+base_line), (255,255,0), -1)
                cv2.putText(bgr_img, text, (l, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

                # print (r-l, b-t, '%.3f'%((r-l)/(b-t)))
            
            for landmark in list_landmarks:
                for i, point in enumerate(list_points):
                    cv2.circle(bgr_img, point, 1, (0, 0, 255), -1) #(147, 112, 219)
            
            if len(list_landmarks) != 0:
                cv2.imshow('lip', img_lip)
                cv2.imshow('face', img_face)
                cv2.rectangle(bgr_img, (roiX1,roiY1), (roiX2,roiY2), (255,255,255), 2) # lip box
            
            cv2.namedWindow('show', 0)
            cv2.imshow('show', bgr_img)
            
            if cv2.waitKey(1) == 27:
                break