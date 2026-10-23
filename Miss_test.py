from pathlib import Path

import cv2
from tqdm import tqdm

import os
import numpy as np
import torch
from utils.img_read_save import img_save,image_read_cv2
import warnings
import logging
import torch.nn.functional as F

from model.Missmodel_inference_acc import missFusionM as tritonM
from model.Missmodel_inference import missFusionM as stdM

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.CRITICAL)



ckpt_path="../ckpt/ep120.pth"
print("\n"*2+"="*80)
ACCELERATION = False
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
if ACCELERATION:
    model =tritonM().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device), strict=False)
    model.eval()
    model.set_acceleration(
        triton_qk=True,
        compact_topk=True,
        triton_bank=True
    )
else:
    model = stdM().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device), strict=False)
    model.eval()


def to_uint8(x):
    """Convert float array to uint8."""
    # x = (x - np.min(x)) / (
    #                 np.max(x) - np.min(x)
    #         )
    x = np.clip(x, 0.0, 1.0)
    return np.uint8(np.round(x * 255.0))

def pad_to_multiple(x, multiple=4):
    _, _, h, w = x.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple

    x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
    return x_pad, (h, w)

def remove_pad(x, orig_size):
    h, w = orig_size
    return x[:, :, :h, :w]


@torch.no_grad()
def predict(tpath,outdirs):
    for img_name in tqdm(os.listdir(os.path.join(tpath,"TIR"))):

            data_TIR = cv2.imread(os.path.join(tpath,"TIR",img_name), cv2.IMREAD_GRAYSCALE).astype(np.float32)
            data_TIR = data_TIR[np.newaxis, np.newaxis, :, :] / 255.

            data_VIS = image_read_cv2(os.path.join(tpath,"VIS",img_name), mode='GRAY')[np.newaxis,np.newaxis, ...]/255.0

            data_TIR, data_VIS = torch.FloatTensor(data_TIR).to(device), torch.FloatTensor(data_VIS).to(device)

            data_TIR, size = pad_to_multiple(data_TIR, 4)
            data_VIS, _ = pad_to_multiple(data_VIS, 4)
            out = model(data_VIS, data_TIR)
            fuse_pred =remove_pad(out, size)
            fuse_np = fuse_pred.squeeze(0).squeeze(0).cpu().numpy()
            fuse_u8 = to_uint8(fuse_np)


            img_bgr = cv2.imread(os.path.join(tpath, "VIS", img_name))
            img_ycbcr = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
            y_orig, cb, cr = cv2.split(img_ycbcr)
            img_ycbcr_new = cv2.merge([fuse_u8, cb, cr])
            img_bgr_new = cv2.cvtColor(img_ycbcr_new, cv2.COLOR_YCrCb2BGR)
            cv2.imwrite(str(outdirs/ f"{img_name.split('.')[0]}.png"), img_bgr_new)

if __name__ == '__main__':

    for dataset in ["TVNF","M3FD", "LLVIP", "FMB"]:
        print(f"Processing {dataset} dataset...")

        tpath = Path(f"../datasets/{dataset}/test")
        outdirs = Path(f"../datasets/{dataset}/results/NiceFusion")
        outdirs.mkdir(parents=True, exist_ok=True)

        predict(tpath,outdirs)