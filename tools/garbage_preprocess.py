# 玻璃杯    27
# 快递纸袋  20
# 塑料     17
# 硬纸板   30
# 金属     33
# 一般垃圾  6
import glob
import os
from shutil import copy, move

# 统计各类数目
# result - {str:str} {pic_name: type}
def counts(pth):
    info = glob.glob(os.path.join(pth, '*.txt'))
    cnt = {}
    for file in info:
        with open(file, "r") as f:
            line = f.readlines()[0]
            fle, idx = line.split(',')
            fle = fle.strip()
            idx = idx.strip()
            if idx in cnt.keys():
                cnt[idx] += 1
            else:
                cnt[idx] = 1

    return cnt

def write_all_info(pth, res, lst):
    basePth = r'D:\Program_self\basicTorch\inputs\garbage-v2\raw'
    info = glob.glob(os.path.join(pth, '*.txt'))
    data = open(res, "w")
    for file in info:
        with open(file, "r") as f:
            line = f.readlines()[0]
            fle, idx = line.split(',')
            fle = fle.strip()
            idx = idx.strip()
            if int(idx) in lst:
                data.write(os.path.join(basePth, fle))
                data.write(' ')
                data.write(idx)
                data.write('\n')

    data.close()

def copyPic(save_pth, lst):
    with open(lst, "r") as f:
        lines = f.readlines()
        for line in lines:
            imgPth, _ = line.split()
            _, img = imgPth.rsplit("\\", 1)
            srcPth = r'E:\dataset\garbage\garbage_classify_v2\train_data_v2'
            srcImg = os.path.join(srcPth, img)
            copy(srcImg, imgPth)

def write_ann(ann_raw):
    pth = r'D:\Program_self\basicTorch\inputs\garbage-v2\raw\ann.txt'
    ann = open(pth, "w")

    with open(ann_raw, "r") as f:
        lines = f.readlines()
        cnt = 1
        for line in lines:
            img, idx = line.split(' ')
            imgName = img.rsplit("\\", 1)[-1]
            ann.write(f'{imgName} {idx}')

    ann.close()

def movePic():
    basePth = r'D:\Program_self\basicTorch\inputs\garbage-v2\raw'
    class_name = {'27':'glass', '20':'bag', '17':'plastic', '30':'cardboard', '33':'metal', '6':'trash'}
    annPth = r'D:\Program_self\basicTorch\inputs\garbage-v2\raw\ann.txt'

    with open(annPth, "r") as f:
        lines = f.readlines()
        for line in lines:
            img, idx = line.split(' ')
            idx = idx.strip()
            srcPth = os.path.join(basePth, img)
            targetPth = os.path.join(basePth, class_name[idx], img)
            move(srcPth, targetPth)

if __name__ == '__main__':
    # pth = r'E:\dataset\garbage\garbage_classify_v2\train_data_v2'
    # save_pth = r'D:\Program_self\basicTorch\inputs\garbage-v2\raw'
    # ann_raw = r'D:\Program_self\basicTorch\inputs\garbage-v2\raw\ann_raw.txt'
    #
    # res = counts(pth)
    # write_all_info(pth, ann_raw, [27, 20, 17, 30, 33, 6])
    # copyPic(save_pth, ann_raw)
    # write_ann(ann_raw)
    movePic()

