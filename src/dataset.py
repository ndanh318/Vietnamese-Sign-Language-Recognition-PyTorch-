import os
import cv2
import math
import random
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

from src.config import CLASSES


class VSLR_Dataset(Dataset):
    def __init__(self, root_path, transform=None, total_image_per_class=500, ratio=0.8, mode="train"):
        self.root_path = root_path
        self.num_class = len(CLASSES)
        self.transform = transform
        if mode == "train":
            self.num_image_per_class = math.ceil(total_image_per_class * ratio)
        else:
            self.num_image_per_class = math.ceil(total_image_per_class * (1 - ratio))
        self.num_sample = self.num_image_per_class * self.num_class

        self.image_paths = []
        self.labels = []
        subdir_paths = []
        for path in os.listdir(self.root_path):
            subdir_paths.append(os.path.join(self.root_path, path))
        for subdir in subdir_paths:
            for idx, image_name in enumerate(os.listdir(subdir)):
                if idx < self.num_image_per_class:
                    image_path = os.path.join(subdir, image_name)
                    self.image_paths.append(image_path)
                    self.labels.append(CLASSES.index(os.path.basename(subdir)))

    def __len__(self):
        return self.num_sample

    def __getitem__(self, item):
        image_path = self.image_paths[item]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image)
        label = self.labels[item]

        return image, label


if __name__ == '__main__':
    root_path = "../dataset/alphabet"
    training_set = VSLR_Dataset(root_path, mode="train")

    # visualize random 20 images
    plt.figure(figsize=(8, 8))
    random_samples = random.sample(range(len(training_set)), 20)
    for idx, random_idx in enumerate(random_samples):
        plt.subplot(4, 5, idx + 1)
        image, label = training_set[random_idx]
        plt.title(CLASSES[label])
        plt.imshow(image)
        plt.axis("off")

    plt.tight_layout()
    plt.savefig("../images/dataset", bbox_inches="tight")
    plt.show()
