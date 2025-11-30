# infer.py
import os
import argparse
import torch
from imageio import imread, imwrite
import cv2
from dataset import ColorizationDataset
from model import UNet
from utils import lab_to_rgb

def load_model(ckpt_path, device):
    model = UNet().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state'] if 'model_state' in ckpt else ckpt)
    model.eval()
    return model

def colorize_image(model, img_path, device, out_path=None):
    img = imread(img_path)
    h, w = 256, 256
    img = cv2.resize(img, (w, h))
    from skimage.color import rgb2lab
    lab = rgb2lab(img).astype('float32')
    L = lab[:,:,0] / 100.0
    L_t = torch.from_numpy(L).unsqueeze(0).unsqueeze(0).float().to(device)
    with torch.no_grad():
        out_ab = model(L_t)
    rgb = lab_to_rgb(L_t.cpu().numpy(), out_ab.cpu().numpy())
    if out_path:
        # save as 0..255 uint8
        imwrite(out_path, (rgb * 255).astype('uint8'))
    return rgb

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--img', required=True)
    p.add_argument('--out', default='out.png')
    args = p.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(args.model, device)
    rgb = colorize_image(model, args.img, device, args.out)
    print("Saved", args.out)
