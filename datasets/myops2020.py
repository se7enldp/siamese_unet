import os
import cv2
import numpy as np
from PIL import Image
from torch.utils import data
from utils import helpers

'''
0 = background 
200 = LV normal myocardium
500 = LV
600 = RV 
1220 = LV myocardial edema
2221 = LV myocardial scars
'''
palette = [[0], [200], [500], [600], [1220], [2221]]

num_classes = 6

# after array_to_img
# c0_lge_t2_mean_std = ((0.164, 0.159), (0.084, 0.066), (0.073, 0.080))
# after to PIL Image
c0_lge_t2_mean_std = ((396.283, 380.769), (230.278, 179.255), (173.758, 182.956))


def get_mean_std(train_loader):
    samples, mean, std = 0, 0, 0
    for (inputs, mask), file_name in train_loader:
        samples += 1
        mean += np.mean(inputs[2].numpy(), axis=(0, 2, 3))
        std += np.std(inputs[2].numpy(), axis=(0, 2, 3))
        print('samples:', samples)

    mean /= samples
    std /= samples
    print(mean, std)


def make_dataset(root, mode, fold):
    assert mode in ['train', 'val', 'test']
    items = []
    if mode == 'train':
        img_path = os.path.join(root, 'Images')
        mask_path = os.path.join(root, 'Labels')

        if 'Augdata' in root:
            data_list = os.listdir(os.path.join(root, 'Labels'))
        else:
            data_list = [l.strip('\n') for l in open(os.path.join(root, 'train1.txt')).readlines()]
        for it in data_list:
            item = ((
                        os.path.join(img_path, 'c0', it),
                        os.path.join(img_path, 'lge', it),
                        os.path.join(img_path, 't2', it)
                    ),
                    os.path.join(mask_path, it))
            items.append(item)
    elif mode == 'val':
        img_path = os.path.join(root, 'Images')
        mask_path = os.path.join(root, 'Labels')
        data_list = [l.strip('\n') for l in open(os.path.join(
            root, 'val{}.txt'.format(fold))).readlines()]
        for it in data_list:
            item = ((
                        os.path.join(img_path, 'c0', it),
                        os.path.join(img_path, 'lge', it),
                        os.path.join(img_path, 't2', it)
                    ),
                    os.path.join(mask_path, it))
            items.append(item)
    else:
        img_path = os.path.join(root, 'Images')
        data_list = [l.strip('\n') for l in open(os.path.join(
            root, 'test.txt')).readlines()]
        for it in data_list:
            item = (
                        os.path.join(img_path, 'c0', it),
                        os.path.join(img_path, 'lge', it),
                        os.path.join(img_path, 't2', it)
                    )
            items.append(item)
    return items


class MyoPS2020(data.Dataset):
    def __init__(self, root, mode, fold, joint_transform=None, roi_crop=None, center_crop=None, transform=None, target_transform=None):
        self.imgs = make_dataset(root, mode, fold)
        self.palette = palette
        self.mode = mode
        if len(self.imgs) == 0:
            raise RuntimeError('Found 0 images, please check the data set')
        self.mode = mode
        self.joint_transform = joint_transform
        self.center_crop = center_crop
        self.roi_crop = roi_crop
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):

        init_size = 0
        if self.mode is not 'test':
            img_path, mask_path = self.imgs[index]
            file_name = mask_path.split('\\')[-1]

            imgs = []
            masks = []
            for path in img_path:
                img = np.load(path)
                mask = np.load(mask_path)
                init_size = mask.shape

                if self.roi_crop:
                    img, mask = self.roi_crop(img, mask)
                img = Image.fromarray(img)
                mask = Image.fromarray(mask)
                imgs.append(img)
                masks.append(mask)

            for i in range(len(imgs)):
                if self.joint_transform is not None:
                    imgs[i], masks[i] = self.joint_transform(imgs[i], masks[i])

                if self.center_crop is not None:
                    imgs[i], masks[i] = self.center_crop(imgs[i], masks[i])

                imgs[i] = np.array(imgs[i])
                imgs[i] = np.expand_dims(imgs[i], axis=2)
                imgs[i] = imgs[i].transpose([2, 0, 1])

                if self.transform is not None:
                    pass
                    imgs[i] = self.transform(imgs[i])
                    # Z-Score
                    imgs[i] = (imgs[i] - c0_lge_t2_mean_std[i][0]) / c0_lge_t2_mean_std[i][1]

            mask = np.array(masks[0])
            mask = np.expand_dims(mask, axis=2)
            mask = helpers.mask_to_onehot(mask, self.palette)
            mask = mask.transpose([2, 0, 1])

            if self.target_transform is not None:
                mask = self.target_transform(mask)

            return (imgs, mask), file_name
        else:
            img_path = self.imgs[index]
            file_name = img_path[0].split('\\')[-1]

            imgs = []
            for path in img_path:
                img = np.load(path)
                init_size = img.shape
                img = Image.fromarray(img)
                imgs.append(img)

            for i in range(len(imgs)):
                if self.joint_transform is not None:
                    imgs[i] = self.joint_transform(imgs[i])

                if self.center_crop is not None:
                    imgs[i] = self.center_crop(imgs[i])

                imgs[i] = np.array(imgs[i])
                imgs[i] = np.expand_dims(imgs[i], axis=2)
                imgs[i] = imgs[i].transpose([2, 0, 1])

                if self.transform is not None:
                    imgs[i] = self.transform(imgs[i])
                    # Z-Score
                    imgs[i] = (imgs[i] - c0_lge_t2_mean_std[i][0]) / c0_lge_t2_mean_std[i][1]

            return imgs, file_name, init_size

    def __len__(self):
        return len(self.imgs)


