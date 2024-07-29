import os

import cv2
import numpy as np
import sklearn.metrics
import torch
import sklearn
import matplotlib.pyplot as plt

from ops.infer.person_attribute_recognition.operator import PersonAttributeRecognitionOperator

THRESH = .50
CROP_IMG_WH_RAT = .5  # width / height
LETTERBOX_COLOR = 0

def parse_pred(fn:str):
    tks = os.path.splitext(fn)[0].split('_')
    prob = float(tks[0].replace('pred', ''))
    camid = int(tks[1])
    date = tks[2]
    time = tks[3]
    return prob, camid, date, time

def __prep(img, dsize=(64, 128)):  # dsize = (width, height)
    return cv2.resize(img, dsize, cv2.INTER_AREA)
    h, w = img.shape[:2]
    
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    #img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    # img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])
    #img = img.reshape(list(img.shape) + [1]).repeat(3, axis=2)
    
    img_wh_rat = w/h
    if img_wh_rat == CROP_IMG_WH_RAT:  # 비율 똑같을 때
        # 어디에 맞추든 상관없음
        interp = cv2.INTER_AREA if h > dsize[1] else cv2.INTER_CUBIC
        return cv2.resize(img, dsize=dsize, interpolation=interp)
    else:
        bg = np.ones((dsize[1], dsize[0], 3), dtype=np.uint8) * LETTERBOX_COLOR
        if img_wh_rat < CROP_IMG_WH_RAT:  # 너비가 좁을 때(높이가 길때) -> 좌우 패딩 부여
            # 상하에 맞춰 리사이즈
            interp = cv2.INTER_AREA if h > dsize[1] else cv2.INTER_CUBIC
            dh = dsize[1]
            dw = (w * dh) // h
            if dw % 2 != 0:
                dw += 1
            resized = cv2.resize(img, (dw, dh), interpolation=interp) 
            cx = dsize[0] // 2
            bg[:, cx - (dw // 2):cx + (dw // 2), :] = resized
        else:  # 너비가 넓을 때(높이가 짧을때) -> 상하 패딩 부여
            # 좌우에 맞춰 리사이즈
            interp = cv2.INTER_AREA if w > dsize[0] else cv2.INTER_CUBIC
            dw = dsize[0]
            dh = (h * dw) // w
            if dh % 2 != 0:
                dh += 1
            resized = cv2.resize(img, (dw, dh), interpolation=interp)
            cy = dsize[1] // 2
            bg[cy - (dh // 2):cy + (dh // 2), :, :] = resized
        return bg


SRC = './240723_test/balanced/'
OUT = f'./240723_test/pred{int(THRESH*100)}_out'

gt_dirs = ['gt_uncls', 'gt_male', 'gt_female']
pred_dirs = ['pred_male', 'pred_female']
for gtd in gt_dirs:
    for predd in pred_dirs:
        os.makedirs(os.path.join(OUT, gtd, predd), exist_ok=True)

male_dpath = os.path.join(SRC, 'male')
female_dpath = os.path.join(SRC, 'female')
uncls_dpath = os.path.join(SRC, 'person')

male_flist = os.listdir(male_dpath)
female_flist = os.listdir(female_dpath)
uncls_flist = os.listdir(uncls_dpath)

male_fpaths = [os.path.join(male_dpath, fn) for fn in male_flist]
female_fpaths = [os.path.join(female_dpath, fn) for fn in female_flist]
uncls_fpaths = [os.path.join(uncls_dpath, fn) for fn in uncls_flist]

cm_id = {
    6002: {
        'TM': 0, 
        'FM': 0, 
        'TF': 0, 
        'FF': 0, 
    },
    6003: {
        'TM': 0, 
        'FM': 0, 
        'TF': 0, 
        'FF': 0, 
    },
    6004: {
        'TM': 0, 
        'FM': 0, 
        'TF': 0, 
        'FF': 0, 
    },
    6005: {
        'TM': 0, 
        'FM': 0, 
        'TF': 0, 
        'FF': 0, 
    }, 
    6008: {
        'TM': 0, 
        'FM': 0, 
        'TF': 0, 
        'FF': 0, 
    }
}


with torch.inference_mode(True):
    opr = PersonAttributeRecognitionOperator()
    opr.load_model()

    male_imgs = []
    male_preped = []
    for male_fpath, male_fname in zip(male_fpaths, male_flist):
        img = cv2.imread(male_fpath)
        preped = __prep(img)
        male_imgs.append(img)
        male_preped.append(preped)
        # cv2.imwrite('./test.png', preped)
    male_out = opr.run(male_preped)

    y_tlist = []
    y_slist = []
    
    for out, img, male_fpath, male_fname in zip(male_out, male_imgs, male_fpaths, male_flist):
        prob = out[22]
        fn = f'pred{prob:.2f}_{male_fname}'
        _, camid, _, _ = parse_pred(fn)
        y_tlist.append(0.)
        y_slist.append(prob)
        if prob < THRESH:
            cm_id[camid]['TM'] += 1
            cv2.imwrite(os.path.join(OUT, 'gt_male', 'pred_male', fn), img)
        else:
            cm_id[camid]['FF'] += 1
            cv2.imwrite(os.path.join(OUT, 'gt_male', 'pred_female', fn), img)

    female_imgs = []
    female_preped = []
    for female_fpath, female_fname in zip(female_fpaths, female_flist):
        img = cv2.imread(female_fpath)
        preped = __prep(img)
        preped = __prep(img)
        female_imgs.append(img)
        female_preped.append(preped)
        # cv2.imwrite('./test.png', preped)
        
    female_out = opr.run(female_preped)
    for out, img, female_fpath, female_fname in zip(female_out, female_imgs, female_fpaths, female_flist):
        prob = out[22]
        fn = f'pred{prob:.2f}_{female_fname}'
        _, camid, _, _ = parse_pred(fn)
        y_tlist.append(1.)
        y_slist.append(prob)
        if prob < THRESH:
            cm_id[camid]['FM'] += 1
            cv2.imwrite(os.path.join(OUT, 'gt_female', 'pred_male', fn), img)
        else:
            cm_id[camid]['TF'] += 1
            cv2.imwrite(os.path.join(OUT, 'gt_female', 'pred_female', fn), img)

    uncls_imgs = []
    uncls_preped = []
    for uncls_fpath, uncls_fname in zip(uncls_fpaths, uncls_flist):
        img = cv2.imread(uncls_fpath)
        preped = __prep(img)
        uncls_imgs.append(img)
        uncls_preped.append(preped)
        # cv2.imwrite('./test.png', preped)
        
    uncls_out = opr.run(uncls_preped)
    for out, img, uncls_fpath, uncls_fname in zip(uncls_out, uncls_imgs, uncls_fpaths, uncls_flist):
        if (prob := out[22]) < THRESH:
            cv2.imwrite(os.path.join(OUT, 'gt_uncls', 'pred_male', f'pred{prob:.2f}_{uncls_fname}'), img)
        else:
            cv2.imwrite(os.path.join(OUT, 'gt_uncls', 'pred_female', f'pred{prob:.2f}_{uncls_fname}'), img)
    
    print(cm_id)
    for k, v in cm_id.items():
        if min(v.values()) > 0:
            acc = (v['TM'] + v['TF']) / (v['TM'] + v['TF'] + v['FM'] + v['FF'])
            m_rec = v['TM'] / (v['TM'] + v['FF'])
            m_prc = v['TM'] / (v['TM'] + v['FM'])
            f_rec = v['TF'] / (v['TF'] + v['FM'])
            f_prc = v['TF'] / (v['TF'] + v['FF'])
            print(f'{acc:.4f},{m_rec:.4f},{m_prc:.4f},{f_rec:.4f},{f_prc:.4f}')
    '''
    recall: GT 중에 맞은거
    precision: PR 중에 맞은거
    '''
    y_tarr = np.array(y_tlist)
    y_sarr = np.array(y_slist)
    out = sklearn.metrics.roc_auc_score(y_true=y_tarr, y_score=y_sarr)
    print(out)
    
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_tarr, y_sarr) 
    roc_auc = sklearn.metrics.auc(fpr, tpr)
    # Plot the ROC curve
    plt.figure()  
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--', label='No Skill')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()