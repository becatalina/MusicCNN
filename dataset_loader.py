import random

import numpy as np
import torch
from PIL import Image
from utils import get_all_files_from_folder,slice_to_categories

from torchvision import transforms


class DatasetLoader(torch.utils.data.Dataset):

    def get_train(self):
        return {'input': sorted(get_all_files_from_folder('../Dataset/cqt')),
                'target': sorted(get_all_files_from_folder('../Dataset/one_hots'))}

    def get_test(self):
        return None

    def get_validation(self):
        return None

    LOADER_TYPES = {'train': get_train, 'test': get_test, 'validation': get_validation}

    def __init__(self, root_dir, transform, set_type='train'):
        self.root_dir = root_dir
        self.transform = transform
        self.data_set = self.LOADER_TYPES[set_type](self)
        print(self[0])

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, idx):
        img_name = self.data_set['input'][idx]
        label_name = self.data_set['target'][idx]

        image = Image.open(img_name)
        target = slice_to_categories(np.load(label_name))

        # image = self.transform(image)

        transform_resize = transforms.Compose([transforms.Resize((224, 224))])
        # target = np.array(transform_resize(target), dtype='int64')
        # target[target == 255] = 0
        return image, target


DatasetLoader("", None)
