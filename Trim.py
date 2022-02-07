import os
import shutil
from tkinter import W
from urllib.error import URLError
import cv2
from cv2 import sort
import numpy as np
import pandas as pd
import ImageProcess as ip
import errno, stat, shutil

def cutImage(image, rect):
    
    x1 = rect[0]
    y1 = rect[1]
    x2 = rect[0] + rect[2]
    y2 = rect[1] + rect[3]
    
    return image[y1:y2, x1:x2]

if __name__ == "__main__":
    
    CWD = 'CPN_Crop_bTrim'
    dst = 'CPN_Crop_Trim'
    
    root_dir = os.path.join(os.getcwd(), 'Ansan', 'data')
    
    if not os.path.exists(os.path.join(root_dir, CWD)):
        raise URLError
    if not os.path.exists(os.path.join(root_dir, dst)):
        os.mkdir(os.path.join(root_dir, dst))
        
    for id in sorted(os.listdir(os.path.join(root_dir, CWD))):
        if not os.path.exists(os.path.join(root_dir, dst, id)):
            os.mkdir(os.path.join(root_dir, dst, id))
            os.mkdir(os.path.join(root_dir, dst, id, 'image'))
            os.mkdir(os.path.join(root_dir, dst, id, 'mask'))
        imgList = [img for img in sorted(os.listdir(os.path.join(root_dir, CWD, id, 'image')))]
        maskList = [img for img in sorted(os.listdir(os.path.join(root_dir, CWD, id, 'mask')))]
        for i in range(len(imgList)):
            imageName = imgList[i]
            maskName = maskList[i]
            if not maskName.split('_')[0] in imageName:
                raise Exception(imageName, maskName)
            image = cv2.imread(os.path.join(root_dir, CWD, id, 'image', imageName), cv2.IMREAD_COLOR)
            mask = cv2.imread(os.path.join(root_dir, CWD, id, 'mask', maskName), cv2.IMREAD_COLOR)
            winname = 'PID: {}, Image 1 {}/{}'.format(id, i+1, len(imgList))
            cv2.namedWindow(winname)
            cv2.moveWindow(winname, 40,30)
            rois = cv2.selectROIs('PID: {}, Image 1 {}/{}'.format(id, i+1, len(imgList)), image)
            cv2.destroyAllWindows()
            
            if len(rois) < 1:
                cv2.imwrite(os.path.join(root_dir, dst, id, 'image', imageName), image)
                cv2.imwrite(os.path.join(root_dir, dst, id, 'mask', maskName), mask)
            else:
                bbox = 0
                for rect in rois:
                    bbox += 1
                    tmpName = imageName.split('.')[0] + '_{}.'.format(bbox) + imageName.split('.')[-1]
                    cv2.imwrite(os.path.join(root_dir, dst, id, 'image', tmpName), cutImage(image, rect))
                    tmpName = maskName.split('.')[0] + '_{}.'.format(bbox) + maskName.split('.')[-1]
                    cv2.imwrite(os.path.join(root_dir, dst, id, 'mask', tmpName), cutImage(mask, rect))
                    