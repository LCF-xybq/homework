import os

def get_file_name(pth):
    filename = os.listdir(pth)

    return filename

def write_ann(img_pth, ann_pth):
    filename = get_file_name(img_pth)
    cnt = len(filename)
    with open(ann_pth, "w") as f:
        for i in range(cnt):
            x = filename[i]
            if x[0] == 'R':
                f.write(f'{x} 1\n')
            elif x[0] == 'O':
                f.write(f'{x} 0\n')

if __name__ == '__main__':
    train_pth = r'D:\Program_self\basicTorch\inputs\garbage\data\train'
    test_pth = r'D:\Program_self\basicTorch\inputs\garbage\data\test'
    train_ann = r'D:\Program_self\basicTorch\inputs\garbage\data\train.txt'
    test_ann = r'D:\Program_self\basicTorch\inputs\garbage\data\test.txt'

    write_ann(train_pth, train_ann)
    write_ann(test_pth, test_ann)