import os
import shutil
from tkinter import W
import cv2
import numpy as np
import pandas as pd
import ImageProcess as ip
import errno, stat, shutil

def splitTwo(image):
    
    w = int(image.shape[1] / 2)
    
    return image[:,0:w], image[:,w:2*w]
    
def splitOne(image):
    
    return image

def splitFour(image):
    
    h = int(image.shape[0]/2)
    w = int(image.shape[1]/2)
    
    return image[0:h,0:w], image[0:h,w:2*w], image[h:2*h,0:w], image[h:2*h,w:2*w]   

if __name__ == "__main__":
    
    Ansan_cts_crop = 'Ansan_median_Crop'
    Ansan_cts_split = 'Ansan_median_Split'
    
    # Set working directory
    cwd_src = Ansan_cts_crop
    cwd_dst = Ansan_cts_split
    
    root_dir = os.path.join(os.getcwd(), 'Ansan', 'data')
    
    PID = 0
    for p_id in sorted(os.listdir(os.path.join(root_dir, cwd_src))):
        PIDLEN = len(os.listdir(os.path.join(root_dir, cwd_src)))
        PID += 1
        if not os.path.exists(os.path.join(root_dir, cwd_dst, p_id)):
            os.mkdir(os.path.join(root_dir, cwd_dst, p_id))
            os.mkdir(os.path.join(root_dir, cwd_dst, p_id, 'image'))
            os.mkdir(os.path.join(root_dir, cwd_dst, p_id, 'mask'))
        else:
            continue
        
        img_list = [img for img in sorted(os.listdir(os.path.join(root_dir, cwd_src, p_id, 'image')))]
        IMGLISTLEN = len(img_list)
        IMGLIST = 0
        for file in img_list:
            IMGLIST += 1
            image_dir = os.path.join(root_dir, cwd_src, p_id, 'image', file)
            mask_dir = os.path.join(root_dir, cwd_src, p_id, 'mask', file.split('.')[0]+'_mask.bmp')
            image = cv2.imread(image_dir, cv2.IMREAD_COLOR)
            mask = cv2.imread(mask_dir, cv2.IMREAD_COLOR)
            
            winname = '{}/{} PID: {}, {}/{} Image: {}'.format(PID, PIDLEN, p_id, IMGLIST, IMGLISTLEN, file)
            cv2.namedWindow(winname)
            cv2.moveWindow(winname, 40, 30)
            cv2.imshow(winname, image)
            key = cv2.waitKey()
            cv2.destroyAllWindows()
            
            src = image_dir
            dst = os.path.join(root_dir, cwd_dst, p_id)
            if key == ord('1'):
                cv2.imwrite(os.path.join(dst, 'mask', file.split('.')[0]+'_mask.bmp'), mask)
                cv2.imwrite(os.path.join(dst, 'image', file), image)                
            elif key == ord('2'):
                imageOne, imageTwo = splitTwo(image)
                maskOne, maskTwo = splitTwo(mask)
                cv2.imwrite(os.path.join(dst, 'mask', file.split('.')[0]+'_mask_1.bmp'), maskOne)
                cv2.imwrite(os.path.join(dst, 'mask', file.split('.')[0]+'_mask_2.bmp'), maskTwo)
                cv2.imwrite(os.path.join(dst, 'image', file.split('.')[0]+'_1.bmp'), imageOne)
                cv2.imwrite(os.path.join(dst, 'image', file.split('.')[0]+'_2.bmp'), imageTwo)                 
            elif key == ord('4'):
                imageOne, imageTwo, imageThree, imageFour = splitFour(image)
                maskOne, maskTwo, maskThree, maskFour = splitFour(mask)
                cv2.imwrite(os.path.join(dst, 'mask', file.split('.')[0]+'_mask_1.bmp'), maskOne)
                cv2.imwrite(os.path.join(dst, 'mask', file.split('.')[0]+'_mask_2.bmp'), maskTwo)
                cv2.imwrite(os.path.join(dst, 'mask', file.split('.')[0]+'_mask_3.bmp'), maskThree)
                cv2.imwrite(os.path.join(dst, 'mask', file.split('.')[0]+'_mask_4.bmp'), maskFour)
                cv2.imwrite(os.path.join(dst, 'image', file.split('.')[0]+'_1.bmp'), imageOne)
                cv2.imwrite(os.path.join(dst, 'image', file.split('.')[0]+'_2.bmp'), imageTwo)
                cv2.imwrite(os.path.join(dst, 'image', file.split('.')[0]+'_3.bmp'), imageThree)
                cv2.imwrite(os.path.join(dst, 'image', file.split('.')[0]+'_4.bmp'), imageFour)                 
            else:
                os.rename(os.path.join(root_dir, cwd_dst, p_id), os.path.join(root_dir, cwd_dst, p_id+'_error'))