import warnings
warnings.filterwarnings("ignore")  # 忽略UserWarning兼容性警告
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

import sys
import numpy as np
sys.path.append('..')

import torch
import torch.nn.functional as F
from losses import L1_Charbonnier_loss,PerceptualLoss,SSIMLoss

from argparse import Namespace
from dataloader import UIEBD_Dataset
from pytorch_lightning import seed_everything
# from torchmetrics.functional import structural_similarity_index_measure as SSIM, peak_signal_noise_ratio as PSNR
from imgqual_utils import PSNR,SSIM
import os

import wandb
#Set seed
seed = 42 #Global seed set to 42
seed_everything(seed)
from pytorch_lightning.loggers import WandbLogger
from HDNet import HDNet

os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

class CoolSystem(pl.LightningModule):
    
    def __init__(self, hparams):
        super(CoolSystem, self).__init__()

        self.params = hparams
            
        self.train_datasets = self.params.train_datasets
        self.train_batchsize = self.params.train_bs
        self.validation_datasets = self.params.val_datasets
        self.val_batchsize = self.params.val_bs
            #Train setting
        self.initlr = self.params.initlr #initial learning
        self.weight_decay = self.params.weight_decay #optimizers weight decay
        self.crop_size = self.params.crop_size #random crop size
        self.num_workers = self.params.num_workers

        self.loss_f = L1_Charbonnier_loss()
        self.ssim_loss = SSIMLoss()

        self.loss_per = PerceptualLoss()

        self.model = HDNet()
        self.save_hyperparameters()


    def forward(self,x_1 ):
        pred,out_head = self.model(x_1)
        return pred,out_head

        
    def configure_optimizers(self):
            # REQUIRED
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.initlr,betas=[0.9,0.999])#,weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,base_lr=self.initlr,max_lr=1.2*self.initlr,cycle_momentum=False)

    
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
            # REQUIRED
        x ,y,id= batch

        b, _, img_h, img_w = x.shape
        img_h_32 = int(32 * np.ceil(img_h / 32.0))
        img_w_32 = int(32 * np.ceil(img_w / 32.0))
        x_pad = F.pad(x, (0, img_w_32 - img_w, 0, img_h_32 - img_h), 'reflect')

        y_hat, out_head = self.forward(x_pad)

        y_hat = y_hat[:, :, :img_h, :img_w]
        out_head = out_head[:, :, :img_h, :img_w]

        loss_d = self.loss_f(y_hat, y)
        loss = loss_d+0.2*self.loss_per(y_hat,y)+ 0.5*self.ssim_loss(y_hat,y)+0.2*self.loss_per(out_head,y)
        self.log('train_loss', loss,sync_dist=True)

        return {'loss': loss}
    
        

    def validation_step(self, batch, batch_idx):
            # OPTIONAL
        x ,y,id= batch
        b, _, img_h, img_w = x.shape
        img_h_32 = int(32 * np.ceil(img_h / 32.0))
        img_w_32 = int(32 * np.ceil(img_w / 32.0))
        x_pad = F.pad(x, (0, img_w_32 - img_w, 0, img_h_32 - img_h), 'reflect')

        y_hat, out_head= self.forward(x_pad)

        y_hat = y_hat[:, :, :img_h, :img_w]
        psnr = PSNR(y_hat, y)
        ssim = SSIM(y_hat, y)
        # psnr = PSNR(y_hat.cpu().clamp(0,1),y.cpu().clamp(0,1),data_range=1).item()
        # ssim = SSIM(y_hat.cpu().clamp(0,1),y.cpu().clamp(0,1),data_range=1).item()

        self.log('psnr', psnr,sync_dist=True)
        self.log('ssim', ssim,sync_dist=True)


        if batch_idx == 0:
            self.logger.experiment.log({
                "compare_image": [wandb.Image(x[0].cpu(), caption=f"raw_image_{id[0]}"),wandb.Image(y[0].cpu(), caption="gt"), wandb.Image(y_hat[0].cpu(), caption="enhancement_pred") ],

            })

                                                    

        return {'psnr': psnr,'ssim':ssim}
        


    def train_dataloader(self):
            # REQUIRED
        train_set = UIEBD_Dataset(self.train_datasets,train=True,size=self.crop_size)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.train_batchsize, shuffle=True, num_workers=self.num_workers)

        return train_loader
        
    def val_dataloader(self):
        val_set = UIEBD_Dataset(self.validation_datasets,train=False)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=self.val_batchsize, shuffle=True, num_workers=self.num_workers)

        return val_loader

def main(hparams):
    model = CoolSystem(hparams)
    check_val = 1
    checkpoint_callback = ModelCheckpoint(
        monitor='psnr',
        filename='B_16_32_64-epoch{epoch:02d}-psnr{psnr:.3f}-ssim{ssim:.3f}',
        auto_insert_metric_name=False,
        every_n_epochs=1,
        save_top_k=3,
        mode="max",
        save_last=True
    )
    RESUME = False
    id = "i7qe8r7a"
    resume_checkpoint_path = fr'underwater_image_enhancement/{id}/checkpoints/last.ckpt'
    name = "HDNet"
    if RESUME:
        logger = WandbLogger(project="underwater_image_enhancement",
                             name=name,
                             log_model=True, resume=True, id=id)
        trainer = pl.Trainer(
            max_epochs=hparams.epochs,
            logger=logger,
            accelerator='cuda',
            callbacks=[checkpoint_callback],
            gradient_clip_val=0.5, gradient_clip_algorithm="value",
            check_val_every_n_epoch=check_val,
            num_sanity_val_steps=0
        )
        trainer.fit(model, ckpt_path=resume_checkpoint_path)
    else:
        logger = WandbLogger(project="underwater_image_enhancement",
                             name=name,
                             log_model=True)
        # logger = TensorBoardLogger("tb_logs", name="my_model")
        trainer = pl.Trainer(
            max_epochs=hparams.epochs,
            logger=logger,
            accelerator='cuda',
            callbacks=[checkpoint_callback],
            gradient_clip_val=0.5, gradient_clip_algorithm="value",
            check_val_every_n_epoch=check_val,
            num_sanity_val_steps=0
        )
        trainer.fit(model)
    

if __name__ == '__main__':
	#your code


    args = {
        'epochs': 400,
        # datasetsw
        'train_datasets': r'/home/xj/datasets/UIEB/train',

        'val_datasets': r'/home/xj/datasets/UIEB/test',
        # bs
        'train_bs': 16,
        # 'train_bs':4,
        'val_bs': 1,
        'initlr': 0.001,  # 0.001
        'weight_decay': 0.01,
        'crop_size': 256,
        'num_workers': 8,
        # Net
        'model_blocks': 5,
        'chns': 64
    }

    hparams = Namespace(**args)

    main(hparams)

    # cool = CoolSystem(hparams).load_from_checkpoint(
    #     "./underwater_image_enhancement/121gbuyg/checkpoints/B_16_32_64-epoch353-psnr23.945-ssim0.924.ckpt")
    # torch.save(cool.model.state_dict(), "checkpoints/best1.ckpt")
    # print("fishish")










