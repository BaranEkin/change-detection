import os
import random
from PIL import Image
import numpy as np

from torch.utils.data import Dataset
from torchvision.transforms import transforms


class ChangeDetDataset(Dataset):
    def __init__(self, pairs_path, reduce_to=None):
        self.reduce_to = reduce_to
        self.transform_train = transforms.Compose([transforms.RandomAffine(0, translate=(.08, .08)),
                                                   transforms.PILToTensor(),
                                                   transforms.RandomErasing(p=1, scale=(.005, .02), value=0),
                                                   transforms.RandomErasing(p=1, scale=(.005, .02), value=0),
                                                   transforms.RandomErasing(p=1, scale=(.005, .02), value=0),
                                                   transforms.RandomErasing(p=1, scale=(.005, .02), value=0),
                                                   transforms.RandomErasing(p=1, scale=(.005, .02), value=0),
                                                   ])
        self.transform_val = transforms.Compose([transforms.PILToTensor()])
        self.data = []
        for folder in os.listdir(pairs_path):
            files = sorted(os.listdir(os.path.join(pairs_path, folder)))
            for i in range(0, len(files) - 1, 2):
                img_path1 = os.path.join(pairs_path, folder, files[i])
                img_path2 = os.path.join(pairs_path, folder, files[i + 1])
                self.data.append([img_path1, img_path2, 0])

        num_positive = len(self.data)
        random.seed(42)
        for i in range(num_positive):
            idx1 = random.randint(0, num_positive - 2)
            idx2 = random.randint(0, num_positive - 2)
            idx3 = random.randint(0, 1)
            idx4 = random.randint(0, 1)

            img_path1 = self.data[idx1][idx3]
            img_path2 = self.data[idx2][idx4]
            self.data.append([img_path1, img_path2, 1])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx, validation=False):

        img1_np = np.array(Image.open(self.data[idx][0]))
        img2_np = np.array(Image.open(self.data[idx][1]))

        if self.reduce_to is not None:
            if validation:
                x = img1_np.shape[0] // 2 - self.reduce_to // 2
                y = img1_np.shape[1] // 2 - self.reduce_to // 2

            else:
                x = random.randint(0, img1_np.shape[0] - self.reduce_to - 1)
                y = random.randint(0, img1_np.shape[1] - self.reduce_to - 1)

            patch1_np = img1_np[x:x + self.reduce_to, y:y + self.reduce_to]
            patch2_np = img2_np[x:x + self.reduce_to, y:y + self.reduce_to]

            img1 = Image.fromarray(patch1_np)
            img2 = Image.fromarray(patch2_np)

            if not validation:
                flip_x = random.randint(0, 3)
                flip_y = random.randint(0, 3)
                rot_90 = random.randint(0, 3)

                if flip_x == 0:
                    img1 = img1.transpose(Image.FLIP_LEFT_RIGHT)
                    img2 = img2.transpose(Image.FLIP_LEFT_RIGHT)
                if flip_y == 0:
                    img1 = img1.transpose(Image.FLIP_TOP_BOTTOM)
                    img2 = img2.transpose(Image.FLIP_TOP_BOTTOM)
                if rot_90 == 0:
                    img1 = img1.transpose(Image.ROTATE_90)
                    img2 = img2.transpose(Image.ROTATE_90)

        else:
            img1 = Image.fromarray(img1_np)
            img2 = Image.fromarray(img2_np)

        transform = self.transform_val if validation else self.transform_train
        return transform(img1), transform(img2), float(self.data[idx][2])


class ChangeDetDatasetReduced(Dataset):
    def __init__(self, base_dataset, validation=False):
        super(ChangeDetDatasetReduced, self).__init__()
        self.base = base_dataset
        self.validation = validation

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        return self.base.dataset.__getitem__(self.base.indices[idx], validation=self.validation)
