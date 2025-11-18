from torch.utils.data import Dataset
from pathlib import Path
from osgeo import gdal
import numpy as np
import torch

def build_image_path(month : int, year : int, region: int, train_test_val: str, data_path : str = 'data/satellite/preprocessed', verbose = False) -> Path :
    path = Path(f"{data_path}/JRC_GSW1_4_MonthlyHistory_{train_test_val}_r{region}/{year}_{str(month).zfill(2)}_01_{train_test_val}_r{region}.tif")
    if verbose: 
        print(path)
    return path

def get_image(month : int, year : int, region: int, train_test_val : str, data_path : str = 'data/satellite/preprocessed') -> np.ndarray :
    with gdal.Open(build_image_path(month, year, region, train_test_val=train_test_val, data_path=data_path)) as img :
        img_array = img.ReadAsArray().astype(np.float32)

    return img_array

class SpacialDataset(Dataset):

    __month = 3

    __train_regions = [r for r in range(1,29)]
    __val_regions = [1]
    __test_regions = [1]

    def __init__(
            self, 
            train_depth : int = 4, 
            pred_depth : int = 1, 
            train_val_test : str = "training",
            data_path : str ="data/satellite/preprocessed",
            device : str = 'cuda:0',
            dtype : type = torch.float32
        ):

        """
        Load the dataset with train / validation / test data from different reach (region) of the river.

        Attributes:
            train_depth: **int** Number images to include in one sample feature set, ie. how many step do we use for prediction
            pred_depth: **int** Number of images to include in the label, ie. how many step do we predict
            train_val_test: **str** Specifies whether to load the 'training', 'testing', or 'validation' set.
        """

        self.train_depth = train_depth
        self.pred_depth = pred_depth
        self.train_val_test = train_val_test
        self.data_path = data_path 
        self.device = device
        self.dtype = dtype

        if train_val_test == 'training':
            self.regions = self.__train_regions
        elif train_val_test == 'validation':
            self.regions = self.__val_regions
        elif train_val_test == 'testing':
            self.regions = self.__test_regions

        self.current_split_data = self.__load_region(self.regions[0])
        self.current_split = 0

        self.samples_per_region = self.__get_samples_per_region()
    
    def __load_region(self, region : int) -> np.ndarray:
        images = get_image(self.__month, 1988, region, self.train_val_test, data_path=self.data_path)[..., np.newaxis]
        for y in range(1989, 2022):
            images = np.concatenate(
                (images, 
                get_image(self.__month, y, region, self.train_val_test, data_path=self.data_path)[...,np.newaxis]), 
                axis=2
            )

        return images

    def __len__(self) -> int:
        return self.samples_per_region * len(self.regions)
    
    def __getitem__(self, index : int):
        if index >= self.__len__():
            raise IndexError()

        if self.__get_region_from_index(index) != self.current_split:
            self.current_split == self.__get_region_from_index(index)
            self.current_split_data = self.__load_region(self.regions[self.current_split])
        
        ts = self.__get_train_start_from_index(index)
        ls = self.__get_label_start_from_index(index)

        return torch.tensor(
            self.current_split_data[:,:,ts:ts + self.train_depth],
            dtype=self.dtype,
            device=self.device
        ), torch.tensor(
            self.current_split_data[:,:,ls:ls+self.pred_depth],
            dtype=self.dtype,
            device=self.device
        )
    
    def __get_region_from_index(self, index):
        return index//self.samples_per_region
    
    def __get_train_start_from_index(self, index):
        return index%self.samples_per_region

    def __get_label_start_from_index(self, index):
        return (index%self.samples_per_region) + self.train_depth
        

    def __get_samples_per_region(self):
        # Number of years - the number of year per sample + 1
        return 2021-1988-self.train_depth-self.pred_depth+1
