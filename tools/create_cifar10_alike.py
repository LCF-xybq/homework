import cv2
import os
import numpy as np

DATA_LEN = 3072  # 32x32x3=3072
CHANNEL_LEN = 1024  # 32x32=1024
SHAPE = (32, 32)  # (160, 90)#32


figure_path = r'D:\Program_self\basicTorch\inputs\garbage\data\test'  # 图片的位置
figure_name_label = r'D:\Program_self\basicTorch\inputs\garbage\data\test.txt'  # 保存图片名称和标签
batch_save = r'D:\Program_self\basicTorch\inputs\garbage\data_32'  # 保存batch文件


## 修改imagelist()标签值
def imread(im_path, shape=None, color="RGB", mode=cv2.IMREAD_UNCHANGED):
    im = cv2.imread(im_path, cv2.IMREAD_UNCHANGED)
    if color == "RGB":
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    if shape != None:
        im = cv2.resize(im, shape)
    return im


def read_data(filename, data_path, shape=None, color='RGB'):
    (shape1, shape2) = shape
    if os.path.isdir(filename):
        print("Can't found data file!")
    else:
        f = open(filename)
        lines = f.read().splitlines()

        count = len(lines)
        print(count)
        data = np.zeros((count, DATA_LEN), dtype=np.uint8)
        lst = [ln.split(' ')[0] for ln in lines]
        label = [int(ln.split(' ')[1]) for ln in lines]

        idx = 0
        c = CHANNEL_LEN
        for ln in lines:
            fname, lab = ln.split(' ')
            # im = imread(os.path.join(data_path, fname), shape=s, color='RGB')
            im = imread(os.path.join(data_path, fname), shape=SHAPE, color='RGB')
            '''
            im = cv2.imread(os.path.join(data_path, fname), cv2.IMREAD_UNCHANGED)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = cv2.resize(im, (s, s))
            '''
            data[idx, :c] = np.reshape(im[:, :, 0], c)
            data[idx, c:2 * c] = np.reshape(im[:, :, 1], c)
            data[idx, 2 * c:] = np.reshape(im[:, :, 2], c)
            label[idx] = int(lab)
            idx = idx + 1

        return data, label, lst


def py2bin(data, label):
    label_arr = np.array(label).reshape(len(label), 1)
    label_uint8 = label_arr.astype(np.uint8)
    arr = np.hstack((label_uint8, data))

    with open(batch_save, 'wb') as f:  # 每个文件夹修改
        # with open('/home/user/PycharmProjects/DataSet_ipanel/layoutdata-160-90/train/train_batch/train_batch_big5small5', 'wb') as f:
        for element in arr.flat:
            f.write(element)


import pickle

BIN_COUNTS = 1  # 每一类的数据为一个batch


def pickled(savepath, data, label, fnames, bin_num=BIN_COUNTS, mode="train", name=None):
    '''
      savepath (str): save path
      data (array): image data, a nx3072 array
      label (list): image label, a list with length n
      fnames (str list): image names, a list with length n
      bin_num (int): save data in several files
      mode (str): {'train', 'test'}
    '''
    assert os.path.isdir(savepath)
    total_num = len(fnames)
    samples_per_bin = total_num // bin_num  # 将/换为// （TypeError: slice indices must be integers or None or have an __index__ method）
    assert samples_per_bin > 0
    idx = 0
    for i in range(bin_num):
        start = i * samples_per_bin
        end = (i + 1) * samples_per_bin

        if end <= total_num:
            dict = {'data': data[start:end, :],
                    'labels': label[start:end],
                    'filenames': fnames[start:end]}
        else:
            dict = {'data': data[start:, :],
                    'labels': label[start:],
                    'filenames': fnames[start:]}
        if mode == "train":
            dict['batch_label'] = "training batch {}".format(name)  # (idx, bin_num)
        else:
            dict['batch_label'] = "testing batch {}".format(name)  # (idx, bin_num)

        with open(os.path.join(savepath, 'data_batch_' + str(name)), 'wb') as fi:  # str(idx)), 'wb') as fi:
            # cPickle.dump(dict, fi)
            pickle.dump(dict, fi)
        # idx = idx + 1


if __name__ == '__main__':
    data_path = figure_path
    file_list = figure_name_label
    save_path = batch_save
    data, label, lst = read_data(file_list, data_path, shape=SHAPE)  # 将图片像素数据转成矩阵和标签列表
    # py2bin(data, label) #将像素矩阵和标签列表转成cifar10 binary version # 二进制版本
    pickled(save_path, data, label, lst, bin_num=1, name='test')