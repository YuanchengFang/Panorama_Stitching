import cv2
import numpy as np
import os
import argparse


# location of the data dir
BASE_DIR = './Homework2'

# locations of target datas
DATA_DIRS = [
    os.path.join(BASE_DIR, 'data%d'%i) for i in [1, 2, 3, 4]
]

# NOTE: Mannually set the reference image
REFRENCES = [
    '113_1301.JPG',
    'IMG_0489.JPG',
    'IMG_0676.JPG',
    None #if no reference, stich images step by step
]

def read_images(datadir='./Homework2/data1', reference=None):
    """Read images

    Parameters
    -------------
    datadir: input data directory
    reference: The filename of the reference image. If None, set no reference.
    
    Return
    -------------
    ref: the reference image, return None if reference is None
    imgs: other images
    """
    ref = None
    imgs = []

    filenames = os.listdir(datadir)
    for filename in filenames:
        img = cv2.imread(os.path.join(datadir, filename))
        if reference == filename:
            ref = img
        else:
            imgs.append(img)
    return ref, imgs


def describe(image, describtor_name='SIFT'):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    if describtor_name.upper() == 'SIFT':
        describtor = cv2.SIFT_create()
    elif describtor_name.upper() == 'ORB':
        # NOTE: if ORB, set features number = 10000
        describtor = cv2.ORB_create(nfeatures=10000)
    else:
        raise NotImplementedError
    kp, des =describtor.detectAndCompute(gray, None)
    return kp, des


def match(des_src, des_dst, matcher_name='BF', rate=0.7):
    if matcher_name.upper() == 'BF':
        matcher = cv2.BFMatcher()
    elif matcher_name.upper() == 'FLANN':
        matcher = cv2.FlannBasedMatcher()
    else:
        raise NotImplementedError
    matches = matcher.knnMatch(des_src, des_dst, k=2)

    func = lambda x : x[0].distance < 0.7*x[1].distance
    matches = list(filter(func, matches))
    matches = [x[0] for x in matches]
    matches = sorted(matches, key = lambda x:x.distance)
    return matches

def homography(matches, kp_src, kp_dst):
    # TODO: match num 100
    good_points = matches #[:100]
    src_pts = np.float32([kp_src[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_dst[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)
    # TODO: cv2.RHO
    M, _ = cv2.findHomography(dst_pts, src_pts, cv2.RHO)
    return M

def mix(src, dst):
    grey_src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    grey_dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    _, mask_src = cv2.threshold(grey_src, 10, 255, cv2.THRESH_BINARY)
    _, mask_dst = cv2.threshold(grey_dst, 10, 255, cv2.THRESH_BINARY)
    mask = cv2.bitwise_and(mask_src, mask_dst)
    mask_inv_src = cv2.bitwise_not(mask_src)
    mask_inv_dst = cv2.bitwise_not(mask_dst)

    src_part = cv2.bitwise_and(src, src, mask=mask_inv_dst)
    dst_part = cv2.bitwise_and(dst, dst, mask=mask_inv_src)

    overlap_src = cv2.bitwise_and(src, src, mask=mask)
    overlap_dst = cv2.bitwise_and(dst, dst, mask=mask)
    
    
    overlap = cv2.addWeighted(overlap_src, 0.7, overlap_dst, 0.3, 1)

    mixed = cv2.add(cv2.add(src_part, dst_part), overlap)
    return mixed

def warpImages(ref, imgs, Hs):
    # TODO: simplify
    h1,w1 = ref.shape[:2]
    xmin, xmax = np.inf, -np.inf
    ymin, ymax = np.inf, -np.inf
    for img2, H in zip(imgs, Hs):

        h2,w2 = img2.shape[:2]
        pts1 = np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
        pts2 = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)
        pts2_ = cv2.perspectiveTransform(pts2, H)
        pts = np.concatenate((pts1, pts2_), axis=0)
        [x_min, y_min] = np.int32(pts.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(pts.max(axis=0).ravel() + 0.5)
        xmin = min(x_min, xmin)
        xmax = max(x_max, xmax)
        ymin = min(y_min, ymin)
        ymax = max(y_max, ymax)

    t = [-xmin,-ymin]
    f = True
    for img2, H in zip(imgs, Hs):
        Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]]) # translate
        if f:
            result = cv2.warpPerspective(img2, Ht.dot(H), (xmax-xmin, ymax-ymin))
            f = False
        else:
            tem = cv2.warpPerspective(img2, Ht.dot(H), (xmax-xmin, ymax-ymin))
            result = mix(result, tem)
    result[t[1]:h1+t[1],t[0]:w1+t[0]] = ref
    return result


def stitch(task_id, describor, matcher):
    print('working at ' + DATA_DIRS[task_id])
    ref, imgs = read_images(datadir = DATA_DIRS[task_id], reference=REFRENCES[task_id])
    kp_ref, des_ref = describe(ref, describor)

    Hs = []
    for img in imgs:
        kp, des = describe(img, describor)
        matches = match(des_ref, des, matcher)
        H = homography(matches, kp_ref, kp)
        Hs.append(H)

    result = warpImages(ref, imgs, Hs)
    if not os.path.exists('./results'):
        os.mkdir('./results')
    cv2.imwrite('./results/%d_%s_%s.jpg'%(task_id+1, describor, matcher), result)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--task', type=int, default=0)
    parser.add_argument('--des', type=str, default='SIFT')
    parser.add_argument('--matcher', type=str, default='BF')

    args = parser.parse_args()

    if args.task == 0:
        for i in range(3):
            stitch(i, args.des, args.matcher)
    elif args.task in range(1, 4):
        stitch(args.task-1, args.des, args.matcher)
    else:
        raise AttributeError
    
    

