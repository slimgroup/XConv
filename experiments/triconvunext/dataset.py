import os
import cv2
import numpy as np
import torch
import torch.utils.data


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        img_ids, 
        img_dir, 
        mask_dir, 
        img_ext, 
        mask_ext, 
        num_classes, 
        transform=None
    ):
        """
        Args:
            img_ids (list): Image ids.
            img_dir: Image file directory.
            mask_dir: Mask file directory.
            img_ext (str): Image file extension.
            mask_ext (str): Mask file extension.
            num_classes (int): Number of classes.
            transform (Compose, optional): Compose transforms of albumentations. Defaults to None.
        
        Note:
            Make sure to put the files as the following structure:
            <dataset name>
            ├── images
            |   ├── 0a7e06.jpg
            │   ├── 0aab0a.jpg
            │   ├── 0b1761.jpg
            │   ├── ...
            |
            └── masks
                ├── 0
                |   ├── 0a7e06.png
                |   ├── 0aab0a.png
                |   ├── 0b1761.png
                |   ├── ...
                |
                ├── 1
                |   ├── 0a7e06.png
                |   ├── 0aab0a.png
                |   ├── 0b1761.png
                |   ├── ...
                ...
        """
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        
        # train_22
        img_id = self.img_ids[idx]

        # (h, w, 3) eg. (522, 775, 3)
        img = cv2.imread(
            os.path.join( # GLAS/train/images/train_22.bmp
                self.img_dir, # GLAS/train/images
                img_id + self.img_ext # train_22.bmp
                )
            )

        # (h, w, 1) eg. (522, 775, 1)
        mask = cv2.imread(
            os.path.join( # GLAS/train/masks/train_22_anno.bmp
                self.mask_dir, 
                img_id + "_anno" + self.mask_ext
            ), 
            cv2.IMREAD_GRAYSCALE
        )[..., None]

        # Convert instance/label map to binary foreground mask
        # (h, w, 1) eg. (522, 775, 1)
        mask = (mask > 0).astype(np.float32)   # 0/1

        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)

            # (h, w, 3) eg. (256, 256, 3)
            img = augmented['image']

            # (h, w, 1) eg. (256, 256, 1)
            mask = augmented['mask']

        # ensure shape + binarity after aug
        if mask.ndim == 2:
            mask = mask[..., None]
            
        mask = (mask > 0.5).astype(np.float32)

        # (c, h, w) eg. (3, 256, 256)
        img = img.transpose(2, 0, 1)

        # (c, h, w) eg. (1, 256, 256)
        mask = mask.transpose(2, 0, 1)

        
        return img, mask, {'img_id': img_id}
