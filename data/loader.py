import os
import cv2
import numpy as np

from torch.utils.data import Dataset


class ImageDataLoader(Dataset):
    def __init__(self, path, transforms=None):
        super().__init__()

        self.path_to_dataset = path
        self.transforms = transforms
        self.image_by_class = []
        self.class_idxs_intervals = {}

        samples_cntr = 0
        dirs_with_images = os.listdir(self.path_to_dataset)
        self.n_classes = len(dirs_with_images)

        for class_dir in sorted(dirs_with_images):
            class_id = int(class_dir) if class_dir.isdigit() else class_dir
            self.class_idxs_intervals[class_id] = [samples_cntr]
            path_to_dir = os.path.join(self.path_to_dataset, class_dir)
            for image in os.listdir(path_to_dir):
                path_to_image = os.path.join(path_to_dir, image)
                if image.split('.')[-1] in ['png', 'jpg', 'jpeg']:
                    self.image_by_class.append((path_to_image, class_id))
                    samples_cntr += 1

            self.class_idxs_intervals[class_id].append(samples_cntr)

        self.image_by_class = np.array(self.image_by_class, dtype=[('filename', '<U128'), ('label', np.int32)])

    def __len__(self):
        return self.image_by_class.shape[0]

    def __getitem__(self, index):
        path_to_image, label = self.image_by_class[index]
        image = self.__load_image(path_to_image)

        return image, label

    def __load_image(self, path_to_image):
        image = cv2.imread(path_to_image)
        if image is None:
            print(f"Cannot find image: {path_to_image}")
            return None
        if image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transforms is not None:
            for t in self.transforms:
                image = t(image=image)['image']

        return image

