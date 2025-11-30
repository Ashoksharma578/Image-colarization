# utils.py
import os
import torch
import numpy as np
from skimage.color import lab2rgb
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import matplotlib.pyplot as plt

def save_checkpoint(state, path):
    torch.save(state, path)

def load_checkpoint(path, model, optimizer=None, map_location=None):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt['model_state'])
    if optimizer and 'optim_state' in ckpt:
        optimizer.load_state_dict(ckpt['optim_state'])
    return ckpt.get('epoch', None), ckpt.get('best_val', None)

def lab_to_rgb(L_tensor, ab_tensor):
    """
    L_tensor: torch tensor (1,H,W) or (B,1,H,W) in 0..1
    ab_tensor: torch tensor (2,H,W) or (B,2,H,W) in -1..1
    Returns numpy RGB image in 0..1
    """
    if torch.is_tensor(L_tensor):
        L = L_tensor.cpu().numpy()
    else:
        L = np.array(L_tensor)
    if torch.is_tensor(ab_tensor):
        ab = ab_tensor.cpu().numpy()
    else:
        ab = np.array(ab_tensor)

    # make shapes (H,W,3)
    if L.ndim == 3:  # B,1,H,W
        L = L[0, 0]
        ab = ab[0].transpose(1, 2, 0)
    elif L.ndim == 2:
        ab = ab.transpose(1, 2, 0)
    else:
        L = L.squeeze()
        ab = ab.squeeze().transpose(1, 2, 0)

    lab = np.zeros((L.shape[0], L.shape[1], 3), dtype=np.float32)
    lab[:,:,0] = L * 100.0
    lab[:,:,1:] = ab * 128.0
    rgb = lab2rgb(lab)  # returns 0..1 float
    return rgb

def compute_metrics(rgb_gt, rgb_pred):
    """
    rgb_gt, rgb_pred: numpy arrays 0..1 shape (H,W,3)
    returns dict with PSNR and SSIM
    """
    # convert to 0..255 for ssim implementation stability
    gt = (rgb_gt * 255).astype('uint8')
    pred = (rgb_pred * 255).astype('uint8')
    psnr = compare_psnr(gt, pred, data_range=255)
    ssim = compare_ssim(gt, pred, multichannel=True, data_range=255)
    return {'psnr': psnr, 'ssim': ssim}

def show_image(img, title=None):
    plt.figure(figsize=(4,4))
    plt.axis('off')
    if title:
        plt.title(title)
    plt.imshow(img)
    plt.show()
