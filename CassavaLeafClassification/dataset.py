from typing import Any, Dict, Optional

import torch
import pytorch_lightning as pl
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
import cv2
import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold



def CassavaDataset(Dataset):
    def __init(self, df:pd.DataFrame, imfolder:str, train:bool=True, transforms=None):
        self.df = df
        self.imfolder = imfolder
        self.train = train
        self.transforms = transforms
        
    def __getitem__(self, index):
        im_path = os.path.join(self.imfolder, self.df.iloc[index]['image_id'])
        x = cv2.imread(im_path, cv2.IMREAD_COLOR)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        
        if self.transforms:
            x = self.transforms(x)['image']
        
        if self.train:
            y = self.df.iloc[index]['label']
            return {
                "x":x,
                "y":y
            }
        else:
            return {"x": x}
    
    def __len__(self):
        return len(self.df)
            

class ClassificationDataModule(pl.LightningDataModule):
    def __init__(
        self,
        images_dir = "",
        train_csv:str="./train.csv",
        batch_size: int = 256,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.train_csv = train_csv
        self.images_dir = images_dir
        ######################
        # data transformations
        ######################
        # References: https://pytorch.org/vision/stable/auto_examples/plot_transforms.html
        

        self.transforms1 = T.RandomApply(
            [
                T.RandomRotation(degrees=(0, 70)),
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=(0.1, 0.6), contrast=1, saturation=0, hue=0.3),
                T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
                T.RandomHorizontalFlip(p=0.3),
            ], 
            p=0.3
        )
        self.train_transforms = T.Compose([
                self.transforms1,
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.valid_transforms = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.train_dataset: Optional[Dataset] = None
        self.valid_dataset: Optional[Dataset] = None


    def prepare_data(self):
        """Download data if needed.
        Do not use it to assign state (self.x = y).
        """
        df = pd.read_csv(self.train_csv)
        df['kfold'] = -1
        df = df.sample(frac=1).reset_index(drop=True)
        stratify = StratifiedKFold(n_splits=5)
        for i, (t_idx, v_idx) in enumerate(stratify.split(X=df['image_id'].values, y=df['label'].values)):
            df.loc[v_idx, "kfold"] = i
        
        df.to_csv("train_folds.csv", index=False)
            
        
    def setup(self, stage: Optional[str] = None):
        """Load data. 
        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        dfx = pd.read_csv("train_folds.csv")
        train = dfx.loc[dfx['kfold'] != 1]
        val = dfx.loc[dfx['kfold'] == 1]
        
        self.train_dataset = CassavaDataset(
            train,
            self.images_dir,
            train=True,
            transforms=self.train_transforms
        )

        self.valid_dataset = CassavaDataset(
            val,
            self.images_dir,
            train=True,
            transforms=self.valid_transforms
        )
        
        
    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.valid_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    
    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass

