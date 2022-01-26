import os
import cv2
import numpy as np
import pandas as pd

def cut_image(image):

    x1 = 0
    y1 = 0
    width = 1024
    height = 768

    return image[y1:y1+height, x1:x1+width]

def draw_Tbbox(image, rois):
    
    raise NotImplementedError

def draw_bbox(image, rois):
    
    for rect in rois:
        print(rois)
        x1, y1, x2, y2 = rect[0], rect[1], rect[2], rect[3]
        image = cv2.rectangle(image, (x1, y1), (x1+x2, y1+y2), (255,0,0), 1)

    winname = 'bbox image'
    cv2.namedWindow(winname)
    cv2.moveWindow(winname, 40,30)
    cv2.imshow('bbox image', image)
    cv2.waitKey(500)
    cv2.destroyAllWindows()
    
    return image

def draw_roi(src, tar, rois):
    '''
    tar: image with rois
    src: original image
    '''
    p_image = tar
    image = src
    for rect in rois:
        x1, y1, x2, y2 = rect[0], rect[1], rect[2], rect[3]
        image[y1:y1+y2+1, x1:x1+x2+1] = p_image[y1:y1+y2+1, x1:x1+x2+1]
 
    winname = 'ROI image'
    cv2.namedWindow(winname)
    cv2.moveWindow(winname, 40,30)       
    cv2.imshow('ROI image', image)
    cv2.waitKey(500)
    cv2.destroyAllWindows()
    
    return image

def draw_mask(image, image_roi):
    
    mask = image_roi - image
    mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    ret, mask_binary = cv2.threshold(mask_gray, 2, 255, 0)
    con, hierarchy = cv2.findContours(mask_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    mask = cv2.drawContours(mask, con, 0, (255, 0, 0), 3)
    mask = np.absolute(255 - mask)
    pts = np.reshape(con[0], (-1, 2))
    #for i in range(len(con) - 1):
    #    tmp = np.reshape(con[i + 1], (-1, 2))
    #    pts = np.append(pts, tmp, axis=0)
     
    #mask = cv2.fillPoly(mask_gray, [pts], (255, 0, 0))     
    kernel_size=7       
    kernel=np.ones((kernel_size,kernel_size),np.uint8)
    th_dil=cv2.dilate(mask_binary,kernel,iterations=1)
    contours, hierarchy = cv2.findContours(th_dil, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        cv2.drawContours(th_dil, [contour], -1, (255, 0, 0), -1)
    final_im=cv2.erode(th_dil,kernel,iterations=1)

    winname = 'mask image'
    cv2.namedWindow(winname)
    cv2.moveWindow(winname, 40,30)    
    cv2.imshow('mask image', final_im)
    cv2.waitKey(500)
    cv2.destroyAllWindows()
    
    return final_im

def process(src=None, tar=None, rois=None):
    '''
    src: original image
    tar: image with roi
    return bbox, roi, mask images
    '''
    bbox = draw_bbox(src, rois)
    roi = draw_roi(tar=tar, src=src, rois=rois)
    mask = draw_mask(image=src, image_roi=tar)

    return bbox, roi, mask    