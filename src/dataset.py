import json
import cv2
import numpy as np
from torch.utils.data import Dataset

class TuSimpleDataset(Dataset):
    def __init__(self, json_path, img_dir, transform=None):
        self.data = [json.loads(line) for line in open(json_path)]
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        img_path = f"{self.img_dir}/{sample['raw_file']}"
        image = cv2.imread(img_path)
        image = cv2.resize(image, (512, 256))

        mask = np.zeros((256, 512), dtype=np.uint8)

        for lane in sample['lanes']:
            for x, y in zip(lane, sample['h_samples']):
                if x != -2:
                    cv2.circle(mask, (x, y), 2, 255, -1)

        mask = cv2.resize(mask, (512, 256))

        image = image / 255.0
        mask = mask / 255.0

        return image.transpose(2, 0, 1), mask[np.newaxis, :]
