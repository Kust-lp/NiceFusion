import random
from pathlib import Path

import cv2
import torch.utils.data as Data
import h5py
import numpy as np
import torch

class H5Dataset(Data.Dataset):
    def __init__(self, h5file_path):
        self.h5file_path = h5file_path
        h5f = h5py.File(h5file_path, 'r')
        self.keys = list(h5f['Tir_patchs'].keys())[:24]
        h5f.close()

    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, index):
        h5f = h5py.File(self.h5file_path, 'r')
        key = self.keys[index]
        TIR = np.array(h5f['Tir_patchs'][key])
        NIR = np.array(h5f['Nir_patchs'][key])
        VIS = np.array(h5f['vis_patchs'][key])
        h5f.close()
        return torch.Tensor(VIS), torch.Tensor(NIR), torch.Tensor(TIR)


class TestDataset(Data.Dataset):
    def __init__(self, path):

        self.root_dir = Path(path)
        self.tir_dir = self.root_dir / "TIR"
        self.nir_dir = self.root_dir / "NIR"
        self.vis_dir = self.root_dir / "VIS"


        self.Timgs = sorted(list(self.tir_dir.glob("*.png")))
        self.Nimgs = sorted(list(self.nir_dir.glob("*.png")))
        self.Vimgs = sorted(list(self.vis_dir.glob("*.png")))


    def __len__(self):
        return len(self.Timgs)

    def __getitem__(self, index):
        tir_p = self.Timgs[index]
        nir_p = self.Nimgs[index]
        vir_p = self.Vimgs[index]

        tIr = np.array(cv2.imread(tir_p, cv2.IMREAD_GRAYSCALE))
        nIr = np.array(cv2.imread(nir_p, cv2.IMREAD_GRAYSCALE))

        vIr =cv2.imread(vir_p).astype('float32')
        vIr = np.round(cv2.cvtColor(vIr, cv2.COLOR_BGR2GRAY))
        tIr_tensor = torch.from_numpy(tIr).float().div(255.0).unsqueeze(0)
        nIr_tensor = torch.from_numpy(nIr).float().div(255.0).unsqueeze(0)
        vIr_tensor = torch.from_numpy(vIr).float().div(255.0).unsqueeze(0)
        return vIr_tensor, nIr_tensor, tIr_tensor, tir_p.name


class BankDataset(Data.Dataset):
    def __init__(self, path, sample_num=300, seed=42):
        self.root_dir = Path(path)
        self.tir_dir = self.root_dir / "TIR"
        self.nir_dir = self.root_dir / "NIR"
        self.vis_dir = self.root_dir / "VIS"

        self.Timgs = sorted(list(self.tir_dir.glob("*.png")))
        self.Nimgs = sorted(list(self.nir_dir.glob("*.png")))
        self.Vimgs = sorted(list(self.vis_dir.glob("*.png")))

        assert len(self.Timgs) == len(self.Nimgs) == len(self.Vimgs)

        total_num = len(self.Timgs)
        sample_num = min(sample_num, total_num)

        rng = random.Random(seed)
        selected_indices = rng.sample(range(total_num), sample_num)

        selected_indices = sorted(selected_indices)

        self.Timgs = [self.Timgs[i] for i in selected_indices]
        self.Nimgs = [self.Nimgs[i] for i in selected_indices]
        self.Vimgs = [self.Vimgs[i] for i in selected_indices]

        self.selected_indices = selected_indices

        print(f"BankDataset: selected {len(self.Timgs)} / {total_num} images for subspace construction.")

    def __len__(self):
        return len(self.Timgs)

    def __getitem__(self, index):
        tir_p = self.Timgs[index]
        nir_p = self.Nimgs[index]
        vir_p = self.Vimgs[index]

        tIr = np.array(cv2.imread(str(tir_p), cv2.IMREAD_GRAYSCALE))
        nIr = np.array(cv2.imread(str(nir_p), cv2.IMREAD_GRAYSCALE))

        vIr = cv2.imread(str(vir_p)).astype("float32")
        vIr = np.round(cv2.cvtColor(vIr, cv2.COLOR_BGR2GRAY))

        tIr_tensor = torch.from_numpy(tIr).float().div(255.0).unsqueeze(0)
        nIr_tensor = torch.from_numpy(nIr).float().div(255.0).unsqueeze(0)
        vIr_tensor = torch.from_numpy(vIr).float().div(255.0).unsqueeze(0)

        return vIr_tensor, nIr_tensor, tIr_tensor