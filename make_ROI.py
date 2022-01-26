import os
import cv2
import numpy as np
import pandas as pd

def draw_Tbbox(image, rois):
    
    raise NotImplementedError

def draw_bbox(image, rois):
    
    for rect in rois:
        x1, y1, x2, y2 = rect[0], rect[1], rect[2], rect[3]
        image = cv2.rectangle(image, (x1, y1), (x1+x2, y1+y2), (255,0,0), 1)
        
    cv2.imshow('bbox image', image)
    cv2.waitKey(500)
    cv2.destroyAllWindows()
    
    return image

def draw_roi(p_image, image, rois):
    
    for rect in rois:
        x1, y1, x2, y2 = rect[0], rect[1], rect[2], rect[3]
        image[y1:y1+y2+1, x1:x1+x2+1] = p_image[y1:y1+y2+1, x1:x1+x2+1]
        
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
    
    
    cv2.imshow('mask image', final_im)
    cv2.waitKey(500)
    cv2.destroyAllWindows()
    
    return final_im

Ansan_cts_raw = 'Ansan_CTS_raw'
Ansan_median_raw = 'Ansan_median_raw'
Test_raw = 'test_raw'

Ansan_cts = 'Ansan_CTS'
Ansan_median = 'Ansan_median'
Test = 'test'

cwd_raw = Test_raw
cwd = Test

#root_dir = 'c:/Users/sdimi/source/repos/Ansan/data'
root_dir = 'c:/Users/singku/google drive ku 175T/code/Ansan/data'
dst = os.path.join(root_dir, cwd)
src = os.path.join(root_dir, cwd_raw)

patient_list = [img for img in sorted(os.listdir(src))]

col = ['patient', 'file', 'width', 'height', 'bbox', 'x1', 'y1', 'x2', 'y2']
ROIS = None

if not os.path.exists(os.path.join(os.getcwd(), 'Ansan', cwd+'_bbox.csv')):
    ROIS = pd.DataFrame(columns=col)
else:
    ROIS = pd.read_csv(os.path.join(os.getcwd(), 'Ansan', cwd+'_bbox.csv'), encoding='utf-8')

for p_id in sorted(os.listdir(src)):
    
    if not os.path.exists(os.path.join(dst, p_id.split('_')[0])):
        os.mkdir(os.path.join(dst, p_id.split('_')[0]))
        
    print('Current patient ID: {}'.format(p_id.split('_')[0]))  
    file_list = [img for img in sorted(os.listdir(os.path.join(src, p_id)))]
    rois = None
    p_rois = None
    p_image = None
    
    for img in sorted(file_list):
        img_dir = os.path.join(src, p_id, img)
        
        # Pass if Bbox (ROI, Mask) images are exist in dst directroy.
        bbox_dst = img.split('.')[0] + '_bbox.' + img.split('.')[-1]
        if os.path.exists(os.path.join(dst, p_id.split('_')[0], bbox_dst)):
            pass
        else:
            pass
        # Korean Path
        image = np.fromfile(img_dir, np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        #image = cv2.imread(img_dir)
        rois = cv2.selectROIs('Patients: {}, {}'.format(p_id, img), image)
        cv2.destroyAllWindows()
        bbox = 0
        width = image.shape[1]
        height =image.shape[0]
        if len(rois) < 1:
            rois = p_rois
            cv2.imwrite(os.path.join(dst, p_id.split('_')[0], img), image)
            
            image_bbox = draw_bbox(image, rois)
            bbox_dst = img.split('.')[0] + '_bbox.' + img.split('.')[-1]
            cv2.imwrite(os.path.join(dst, p_id.split('_')[0], bbox_dst), image_bbox)
            
            image_roi = draw_roi(p_image, image, rois)
            roi_dst = img.split('.')[0] + '_roi.' + img.split('.')[-1]
            cv2.imwrite(os.path.join(dst, p_id.split('_')[0], roi_dst), image_roi)
            
            image = np.fromfile(img_dir, np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            image_mask = draw_mask(image=image, image_roi=image_roi)
            mask_dst = img.split('.')[0] + '_mask.' + img.split('.')[-1]
            cv2.imwrite(os.path.join(dst, p_id.split('_')[0], mask_dst), image_mask)
            
            for rect in rois:
                bbox += 1
                x1 = rect[0]
                y1 = rect[1]    
                x2 = rect[0] + rect[2]
                y2 = rect[1] + rect[3]
                new_row = {'patient' : p_id.split('_')[0], 
                        'file' : img.split('.')[0], 
                        'width' : width, 
                        'height' : height,
                        'bbox' : bbox,
                        'x1' : x1, 'y1' : y1, 'x2' : x2, 'y2' : y2}
                print(new_row)
                ROIS = ROIS.append(new_row, ignore_index=True)            
        else:
            p_rois = rois
            p_image = image
            
print('Save path: ', os.path.join(os.getcwd(), 'Ansan', 'Ansan_CTS_'+'bbox.csv'))
ROIS.to_csv(os.path.join(os.getcwd(), 'Ansan','Ansan_CTS_'+'bbox.csv'), encoding='utf-8')