import argparse
import os

import torch
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from torchmetrics.functional import structural_similarity_index_measure as SSIM, peak_signal_noise_ratio as PSNR
import warnings
from torchvision.utils import save_image,make_grid
from tqdm import tqdm
from torch.utils.data import DataLoader
from myutils.dataloader import UIEBD_Dataset
from HDNet import HDNet
from PIL import Image
from PIL import Image, ImageEnhance
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_type', type=str, default='uieb')
parser.add_argument('--dataset', type=str, default='/home/xj/datasets/UIEB/test', help='path of UIED')
parser.add_argument('--savepath', type=str, default='/home/xj/datasets/UIEB/pred2', help='path of output image')
parser.add_argument('--model_path', type=str, default=r'/home/xj/code/python/enhance/FiveAPlus-Network-main/checkpoints/best_c_1.ckpt', help='path of FA+Net checkpoint')
opt = parser.parse_args()

val_set = UIEBD_Dataset(opt.dataset,train=False)
val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4)


netG_1 = HDNet().cuda()

if __name__ == '__main__':   

    ssim = []
    psnr = []
    mae = []
    
    g1ckpt1 = opt.model_path
    ckpt = torch.load(g1ckpt1)
    netG_1.load_state_dict(ckpt)
    netG_1.eval()
   
    savepath_dataset = os.path.join(opt.savepath,opt.dataset_type)
    if not os.path.exists(savepath_dataset):
        os.makedirs(savepath_dataset)
    loop = tqdm(enumerate(val_loader),total=len(val_loader))
    name = 0
    for idx,(raw,gt,id) in loop:
        
        with torch.no_grad():
                
                raw = raw.cuda()
                gt = gt.cuda()
                b, _, img_h, img_w = raw.shape
                img_h_32 = int(32 * np.ceil(img_h / 32.0))
                img_w_32 = int(32 * np.ceil(img_w / 32.0))
                raw_pad = F.pad(raw, (0, img_w_32 - img_w, 0, img_h_32 - img_h), 'reflect')

                enhancement_img,enhancementhead= netG_1(raw_pad)

                y_pred = enhancement_img[:, :, :img_h, :img_w]

                ######后处理######## ：
                # y_pred_ndarr = make_grid(y_pred).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
                #
                # y_pred= Image.fromarray(y_pred_ndarr)
                #
                # contrasted = ImageEnhance.Contrast(y_pred)
                # contrasted = contrasted.enhance(1)
                #
                # colored = ImageEnhance.Color(contrasted)
                # y_pred = colored.enhance(1)
                #
                # y_pred = transforms.ToTensor()(y_pred)[None].cuda()
                ######后处理########

                psnr.append(PSNR(y_pred.clamp(0,1), gt, data_range=1 - 0).item())
                ssim.append(SSIM(y_pred.clamp(0,1), gt, data_range=1 - 0).item())
                # np.mean(np.absolute((imgA / 255.0 - imgB / 255.0)))
                mae.append(np.mean(np.absolute(y_pred.clamp(0,1).cpu().numpy()-gt.clamp(0,1).cpu().numpy())))

                save_image(y_pred.clamp(0,1),os.path.join(savepath_dataset,'%s'%(id)),normalize=False)
    print("psnr:",np.array(psnr).mean())
    print("ssim:",np.array(ssim).mean())
    print("mae:", np.array(mae).mean())