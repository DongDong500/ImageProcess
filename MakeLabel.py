import os
import shutil
import cv2
import numpy as np
import pandas as pd
import ImageProcess as ip
import errno, stat, shutil

def handleRemoveReadonly(func, path, exc):
    excvalue = exc[1]
    if func in (os.rmdir, os.remove) and excvalue.errno == errno.EACCES:
        os.chmod(path, stat.S_IRWXU| stat.S_IRWXG| stat.S_IRWXO) # 0777
        func(path)
    else:
        raise

if __name__ == "__main__":
    
    Ansan_cts_raw = 'Ansan_CTS_raw'
    Ansan_median_raw = 'Ansan_median_raw'
    Test_raw = 'test_raw'

    Ansan_cts = 'Ansan_CTS'
    Ansan_median = 'Ansan_median'
    Test = 'test'
    
    # Set working directory
    cwd_raw = Ansan_median_raw
    cwd = Ansan_median 
    
    
    root_dir = os.path.join(os.getcwd(), 'Ansan', 'data')
    #root_dir = 'c:/Users/sdimi/source/repos/Ansan/data'
    #root_dir = 'c:/Users/singku/google drive ku 175T/code/Ansan/data'
    dst = os.path.join(root_dir, cwd)
    src = os.path.join(root_dir, cwd_raw)

    col = ['patient', 'file', 'width', 'height', 'bbox', 'x1', 'y1', 'x2', 'y2']
    ROIS = None
    
    # Read csv file if exists
    if not os.path.exists(os.path.join(os.getcwd(), 'Ansan', cwd+'_bbox.csv')):
        ROIS = pd.DataFrame(columns=col)
    else:
        ROIS = pd.read_csv(os.path.join(os.getcwd(), 'Ansan', cwd+'_bbox.csv'), encoding='utf-8', index_col = 0)

    for p_id in sorted(os.listdir(src)):
        
        if len(p_id.split('_')) > 1:
            os.rename(os.path.join(src, p_id), os.path.join(src, p_id.split('_')[0]))
            p_id = p_id.split('_')[0]

        if not os.path.exists(os.path.join(dst, p_id)):
            os.mkdir(os.path.join(dst, p_id))
            os.mkdir(os.path.join(dst, p_id, 'bbox'))
            os.mkdir(os.path.join(dst, p_id, 'roi'))
            os.mkdir(os.path.join(dst, p_id, 'mask'))
            os.mkdir(os.path.join(dst, p_id, 'image'))
        
        img_list = [img for img in sorted(os.listdir(os.path.join(src, p_id)))]
        dicts = []

        if len(img_list) % 2 == 1:
            print('Not valid images are in {}'.format(p_id.split('_')[0]))
            print('Dir: {}'.format(src))
            new_row = None
        else:
            for i in range(int(len(img_list)/2)):
                img1_dir = os.path.join(src, p_id, img_list[2*i])
                img2_dir = os.path.join(src, p_id, img_list[2*i+1])
                bbox_n = 0

                if os.path.exists(os.path.join(dst, p_id, 'image', img_list[2*i])):
                    continue
                if os.path.exists(os.path.join(dst, p_id, 'image', img_list[2*i+1])):
                    continue

                try:
                    img1 = cv2.imread(img1_dir, cv2.IMREAD_COLOR)
                    img2 = cv2.imread(img2_dir, cv2.IMREAD_COLAR)
                    if img1 is None or img2 is None:
                        raise FileNotFoundError
                except: # Korean path
                    img1 = np.fromfile(img1_dir, np.uint8)
                    img1 = cv2.imdecode(img1, cv2.IMREAD_COLOR)
                    img2 = np.fromfile(img2_dir, np.uint8)
                    img2 = cv2.imdecode(img2, cv2.IMREAD_COLOR)
                
                img1 = ip.cut_image(img1)
                img2 = ip.cut_image(img2)
                
                width = img1.shape[1]
                height =img2.shape[0]

                winname = 'PID: {}, Image 1 {}/{}'.format(p_id, 2*i+1, len(img_list))
                cv2.namedWindow(winname)
                cv2.moveWindow(winname, 40,30)
                rois1 = cv2.selectROIs('PID: {}, Image 1 {}/{}'.format(p_id, 2*i+1, len(img_list)), img1)
                cv2.destroyAllWindows()
                winname = 'PID: {}, Image 2 {}/{}'.format(p_id, 2*i+2, len(img_list))
                cv2.namedWindow(winname)
                cv2.moveWindow(winname, 40,30)                
                rois2 = cv2.selectROIs('PID: {}, Image 2 {}/{}'.format(p_id, 2*i+2, len(img_list)), img2)
                cv2.destroyAllWindows()

                if len(rois1) >= 1 and len(rois2) < 1:
                    src_i = img2
                    tar_i = img1
                    img = img_list[i*2+1]
                    rois = rois1
                    # (0) Save original image
                    cv2.imwrite(os.path.join(dst, p_id, 'image', img), src_i)
                    # (1) Save bbox image
                    bbox = ip.draw_bbox(img2, rois)
                    bbox_dst = img.split('.')[0] + '_bbox.' + img.split('.')[-1]
                    cv2.imwrite(os.path.join(dst, p_id, 'bbox', bbox_dst), bbox)
                    # (2) Save roi image
                    roi = ip.draw_roi(src=img2, tar=img1, rois=rois)
                    roi_dst = img.split('.')[0] + '_roi.' + img.split('.')[-1]
                    cv2.imwrite(os.path.join(dst, p_id, 'roi', roi_dst), roi)
                    # (3) Save mask image
                    #     Need to reload new original image to make mask image
                    image = np.fromfile(img2_dir, np.uint8)
                    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
                    image = ip.cut_image(image)
                    mask = ip.draw_mask(image=image, image_roi=roi)
                    mask_dst = img.split('.')[0] + '_mask.' + img.split('.')[-1]
                    cv2.imwrite(os.path.join(dst, p_id, 'mask', mask_dst), mask)
                elif len(rois1) < 1 and len(rois2) >= 1:
                    src_i = img1
                    tar_i = img2
                    img = img_list[i*2]
                    rois = rois2
                    # (0) Save original image
                    cv2.imwrite(os.path.join(dst, p_id, 'image', img), src_i)
                    # (1) Save bbox image
                    bbox = ip.draw_bbox(img1, rois)
                    bbox_dst = img.split('.')[0] + '_bbox.' + img.split('.')[-1]
                    cv2.imwrite(os.path.join(dst, p_id, 'bbox', bbox_dst), bbox)
                    # (2) Save roi image
                    roi = ip.draw_roi(src=img1, tar=img2, rois=rois)
                    roi_dst = img.split('.')[0] + '_roi.' + img.split('.')[-1]
                    cv2.imwrite(os.path.join(dst, p_id, 'roi', roi_dst), roi)
                    # (3) Save mask image
                    #     Need to reload new original image to make mask image
                    image = np.fromfile(img1_dir, np.uint8)
                    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
                    image = ip.cut_image(image)
                    mask = ip.draw_mask(image=image, image_roi=roi)
                    mask_dst = img.split('.')[0] + '_mask.' + img.split('.')[-1]
                    cv2.imwrite(os.path.join(dst, p_id, 'mask', mask_dst), mask)
                else:
                    dicts = None
                    break
                for rect in rois:
                    bbox_n += 1
                    x1, y1, x2, y2 = rect[0], rect[1], rect[0]+rect[2], rect[1]+rect[3]
                    new_row = {'patient' : p_id, 
                                'file' : img.split('.')[0], 
                                'width' : width, 
                                'height' : height,
                                'bbox' : bbox_n,
                                'x1' : x1, 'y1' : y1, 'x2' : x2, 'y2' : y2}
                    dicts.append(new_row)
            new_row = dicts

        if new_row is None:
            if not os.path.exists(os.path.join(dst, p_id+'_error')):
                os.rename(os.path.join(dst, p_id), os.path.join(dst, p_id+'_error'))
            else:
                try:
                    os.rmdir(os.path.join(dst, p_id+'_error'))
                except:
                    shutil.rmtree(os.path.join(dst, p_id+'_error', 'image'), ignore_errors=False, onerror=handleRemoveReadonly)
                    shutil.rmtree(os.path.join(dst, p_id+'_error', 'roi'), ignore_errors=False, onerror=handleRemoveReadonly)
                    shutil.rmtree(os.path.join(dst, p_id+'_error', 'bbox'), ignore_errors=False, onerror=handleRemoveReadonly)
                    shutil.rmtree(os.path.join(dst, p_id+'_error', 'mask'), ignore_errors=False, onerror=handleRemoveReadonly)
                    shutil.rmtree(os.path.join(dst, p_id+'_error'), ignore_errors=False, onerror=handleRemoveReadonly)
                os.rename(os.path.join(dst, p_id), os.path.join(dst, p_id+'_error'))
        else:
            if len(new_row) > 0:
                print(new_row)
                ROIS = ROIS.append(new_row, ignore_index=True)
                print('Save path: ', os.path.join(os.getcwd(), 'Ansan', cwd+'_bbox.csv'))
                ROIS.to_csv(os.path.join(os.getcwd(), 'Ansan', cwd+'_bbox.csv'), encoding='utf-8')
          