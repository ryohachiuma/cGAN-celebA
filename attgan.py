import os
import random
import numpy as np
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from nn import LinearBlock, Conv2dBlock, ConvTranspose2dBlock
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
from PIL import Image, ImageFilter
import torch.autograd as autograd
import torch.nn.functional as F
import torch.optim as optim
import argparse
from torchsummary import summary
from addict import Dict
import yaml

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='./config/test.yml', help='the path to config yaml file')
args = parser.parse_args()

CONFIG = Dict(yaml.safe_load(open(args.config)))
logdir = os.path.join('./log', CONFIG.log_name)
if not os.path.isdir(logdir):
    os.makedirs(logdir)
    os.system('cp %s %s' % (args.config, logdir))


"""setup"""
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


class Mydatasets(torch.utils.data.Dataset):
        def __init__(self, dir_path='./dataset', transform = None):
            self.transform = transform
            self.data = []
            self.dir_path = dir_path
            self.attr = []
            attr_data = np.genfromtxt(os.path.join(self.dir_path, 'list_attr_celeba.csv'), dtype=str, delimiter=',')
            self.img_names = attr_data[1:, 0]
            attr_list = attr_data[0,  1:]
            attr_data = attr_data[1:, 1:]
            selected_attr = [attr_list.tolist().index(att) for att in CONFIG.attribute]
            selected_attr_data = attr_data[:, selected_attr].astype(np.float32)
            selected_attr_data[selected_attr_data == -1] = 0 # might not be necessary
            self.attr = torch.from_numpy(selected_attr_data)
            
        def __len__(self):
            return len(self.attr)

        def __getitem__(self, idx):
            img = Image.open(os.path.join(self.dir_path, self.img_names[idx]))
            out_attr = self.attr[idx]
            if self.transform:
                img = self.transform(img)
            return img, out_attr
        

class Generator(torch.nn.Module):
    def __init__(self, n_in = 1024, max_dim = 1024,
                 dec_dim=64, dec_layers=5, dec_norm_fn='batchnorm', dec_acti_fn='lrelu',
                 shortcut_layers=0, inject_layers=0, img_size=128, attribute_num = 0):
        super(Generator, self).__init__()
        
        self.shortcut_layers = min(shortcut_layers, dec_layers - 1)
        self.inject_layers = min(inject_layers, dec_layers - 1)
        self.f_size = img_size // 2**dec_layers  # f_size = 4 for 128x128
        
        layers = []
        n_in = n_in + attribute_num  # 1024 + 40
        for i in range(dec_layers):
            if i < dec_layers - 1:
                n_out = min(dec_dim * 2**(dec_layers-i-1), max_dim)
                layers += [ConvTranspose2dBlock(
                    n_in, n_out, (4, 4), stride=2, padding=1, norm_fn=dec_norm_fn, acti_fn=dec_acti_fn
                )]
                n_in = n_out
                n_in = n_in + n_in//2 if self.shortcut_layers > i else n_in
                n_in = n_in + attribute_num if self.inject_layers > i else n_in
            else:
                layers += [ConvTranspose2dBlock(
                    n_in, 3, (4, 4), stride=2, padding=1, norm_fn='none', acti_fn='tanh'
                )]
        self.dec_layers = torch.nn.ModuleList(layers)


    def decode(self, zs, a):
        a_tile = a.view(a.size(0), -1, 1, 1).repeat(1, 1, self.f_size, self.f_size)
        z = torch.cat([zs, a_tile], dim=1)
        for i, layer in enumerate(self.dec_layers):
            z = layer(z)
            if self.shortcut_layers > i:  # Concat 1024 with 512
                z = torch.cat([z, zs[len(self.dec_layers) - 2 - i]], dim=1)
            if self.inject_layers > i:
                a_tile = a.view(a.size(0), -1, 1, 1) \
                          .repeat(1, 1, self.f_size * 2**(i+1), self.f_size * 2**(i+1))
                z = torch.cat([z, a_tile], dim=1)
        return z
    
    def forward(self, x, a):
        return self.decode(x, a)


class Discriminators(torch.nn.Module):
    # No instancenorm in fcs in source code, which is different from paper.
    def __init__(self, dim=64, norm_fn='instancenorm', acti_fn='lrelu',
                 fc_dim=1024, max_dim=1024, fc_norm_fn='none', fc_acti_fn='lrelu', n_layers=5, img_size=128):
        super(Discriminators, self).__init__()
        self.f_size = img_size // 2**n_layers
        layers = []
        n_in = 3
        for i in range(n_layers):
            n_out = min(dim * 2**i, max_dim)
            layers += [Conv2dBlock(
                n_in, n_out, (4, 4), stride=2, padding=1, norm_fn=norm_fn, acti_fn=acti_fn
            )]
            n_in = n_out
        self.conv = torch.nn.Sequential(*layers)
        self.fc_adv = torch.nn.Sequential(
            LinearBlock(n_out * self.f_size * self.f_size, fc_dim, fc_norm_fn, fc_acti_fn),
            LinearBlock(fc_dim, 1, 'none', 'none')
        )
    
    def forward(self, x):
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        return self.fc_adv(h)

class cGAN():
    def __init__(self, device):
        self.mode = CONFIG.gan_mode
        self.lambda_1, self.lambda_2, self.lambda_gp = CONFIG.loss_weights
        self.device = device
        self.noise_size = CONFIG.noise_size
        self.attribute_num = len(CONFIG.attribute)

        self.G = Generator(n_in = CONFIG.noise_size, max_dim = CONFIG.max_dim, dec_layers=CONFIG.num_layers, shortcut_layers=CONFIG.shortcut_layers, \
            img_size=CONFIG.img_size, attribute_num=self.attribute_num)
        self.G.train()
        self.f_size = self.G.f_size
        self.G.cuda()

        
        self.D = Discriminators(fc_dim=CONFIG.noise_size, max_dim=CONFIG.max_dim, n_layers=CONFIG.num_layers, img_size=CONFIG.img_size)
        self.D.train()
        self.D.cuda()


        self.optim_G = optim.Adam(self.G.parameters(), lr=CONFIG.lr_G, betas=(0.5, 0.999))
        self.optim_D = optim.Adam(self.D.parameters(), lr=CONFIG.lr_D, betas=(0.5, 0.999))
    
    def set_lr(self, lr):
        for g in self.optim_G.param_groups:
            g['lr'] = lr
        for g in self.optim_D.param_groups:
            g['lr'] = lr
    
    def trainG(self, img, att):
        for p in self.D.parameters():
            p.requires_grad = False
        
        batch_size = img.size(0)
        noise = torch.randn(batch_size, self.noise_size, self.f_size, self.f_size, device=self.device)
        img_recon = self.G(noise, att)
        d_fake = self.D(img_recon)
        
        if self.mode == 'wgan':
            gf_loss = -d_fake.mean()
        if self.mode == 'lsgan':  # mean_squared_error
            gf_loss = F.mse_loss(d_fake, torch.ones_like(d_fake))
        if self.mode == 'dcgan':  # sigmoid_cross_entropy
            gf_loss = F.binary_cross_entropy_with_logits(d_fake, torch.ones_like(d_fake))
        gr_loss = F.l1_loss(img_recon, img)
        g_loss = gf_loss + self.lambda_1 * gr_loss
        
        self.optim_G.zero_grad()
        g_loss.backward()
        self.optim_G.step()
        
        errG = {
            'g_loss': g_loss.item(), 'gf_loss': gf_loss.item(),
            'gr_loss': gr_loss.item()
        }
        return errG

    def predict(self, att):

        batch_size = att.size(0)

        noise = torch.randn(batch_size, self.noise_size, self.f_size, self.f_size, device=self.device)
        img_recon = self.G(noise, att)
        return img_recon
    
    def trainD(self, img, att):
        for p in self.D.parameters():
            p.requires_grad = True
        
        batch_size = img.size(0)
        noise = torch.randn(batch_size, self.noise_size, self.f_size, self.f_size, device=self.device)
        img_fake = self.G(noise, att).detach()
        d_real = self.D(img)
        d_fake = self.D(img_fake)
        
        def gradient_penalty(f, real, fake=None):
            def interpolate(a, b=None):
                if b is None:  # interpolation in DRAGAN
                    beta = torch.rand_like(a)
                    b = a + 0.5 * a.var().sqrt() * beta
                alpha = torch.rand(a.size(0), 1, 1, 1)
                alpha = alpha.cuda()
                inter = a + alpha * (b - a)
                return inter
            x = interpolate(real, fake).requires_grad_(True)
            pred = f(x)
            if isinstance(pred, tuple):
                pred = pred[0]
            grad = autograd.grad(
                outputs=pred, inputs=x,
                grad_outputs=torch.ones_like(pred),
                create_graph=True, retain_graph=True, only_inputs=True
            )[0]
            grad = grad.view(grad.size(0), -1)
            norm = grad.norm(2, dim=1)
            gp = ((norm - 1.0) ** 2).mean()
            return gp
        
        if self.mode == 'wgan':
            wd = d_real.mean() - d_fake.mean()
            df_loss = -wd
            df_gp = gradient_penalty(self.D, img, img_fake)
        if self.mode == 'lsgan':  # mean_squared_error
            df_loss = F.mse_loss(d_real, torch.ones_like(d_fake)) + \
                      F.mse_loss(d_fake, torch.zeros_like(d_fake))
            df_gp = gradient_penalty(self.D, img)
        if self.mode == 'dcgan':  # sigmoid_cross_entropy
            df_loss = F.binary_cross_entropy_with_logits(d_real, torch.ones_like(d_real)) + \
                      F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake))
            df_gp = gradient_penalty(self.D, img)
        d_loss = df_loss + self.lambda_gp * df_gp
        
        self.optim_D.zero_grad()
        d_loss.backward()
        self.optim_D.step()
        
        errD = {
            'd_loss': d_loss.item(), 'df_loss': df_loss.item(), 
            'df_gp': df_gp.item()
        }
        return errD
    
    def train(self):
        self.G.train()
        self.D.train()
    
    def eval(self):
        self.G.eval()
        self.D.eval()
    
    def save(self, path):
        states = {
            'G': self.G.state_dict(),
            'D': self.D.state_dict(),
            'optim_G': self.optim_G.state_dict(),
            'optim_D': self.optim_D.state_dict()
        }
        torch.save(states, path)
    
    def load(self, path):
        states = torch.load(path, map_location=lambda storage, loc: storage)
        if 'G' in states:
            self.G.load_state_dict(states['G'])
        if 'D' in states:
            self.D.load_state_dict(states['D'])
        if 'optim_G' in states:
            self.optim_G.load_state_dict(states['optim_G'])
        if 'optim_D' in states:
            self.optim_D.load_state_dict(states['optim_D'])
    
    def saveG(self, path):
        states = {
            'G': self.G.state_dict()
        }
        torch.save(states, path)

def main():
    set_seed(CONFIG.seed)
    transform=transforms.Compose([
            transforms.CenterCrop(CONFIG.crop_size),
            transforms.Resize((CONFIG.img_size, CONFIG.img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                             ])
    
    dataset = Mydatasets(transform=transform)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=CONFIG.batch_size,
                                         shuffle=True, num_workers=int(24))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    writer = SummaryWriter(log_dir = logdir)
    model = cGAN(device=device)


    # Generate attribute pattern
    combination_num = 2**(len(CONFIG.attribute))
    att_pattern = [np.binary_repr(idx, width=len(CONFIG.attribute)) for idx in range(combination_num)]
    att_list = np.asarray([[int(idx) for idx in pattern] for pattern in att_pattern])
    att_list = np.tile(att_list, (CONFIG.vis_num, 1))
    att_tensor = torch.from_numpy(att_list.astype(np.float32)).to(device)

    # traning loop
    for epoch in range(CONFIG.max_epoch):
        model.train()
        for itr, data in enumerate(dataloader):
            real_image = data[0].to(device)
            real_label = data[1].to(device)
            if (itr+1) % (CONFIG.disc_epoch+1) != 0:
                errD = model.trainD(real_image, real_label)
                print('[{}/{}][{}/{}] Loss_D: {:.7f}'
                      .format(epoch + 1, CONFIG.max_epoch,
                              itr + 1, len(dataloader), errD['d_loss']))
            else:
                errG = model.trainG(real_image, real_label)
                print('[{}/{}][{}/{}] Loss_G: {:.7f}'
                      .format(epoch + 1, CONFIG.max_epoch,
                              itr + 1, len(dataloader), errG['g_loss']))
        
        if epoch % CONFIG.save_interval == 0:
            epoch_dir = os.path.join(logdir, 'epoch_%04d' % (epoch))
            if not os.path.isdir(epoch_dir):
                os.makedirs(epoch_dir)
            save_path = os.path.join(epoch_dir, 'model.pth')
            model.save(save_path)
            model.eval()
            with torch.no_grad():
                recon_image = model.predict(att_tensor)   
                vutils.save_image(recon_image.detach(), os.path.join(epoch_dir, 'img.png'), normalize=True, nrow=CONFIG.vis_num)


    # グラフ作成
    writer.close()

if __name__ == '__main__':
    main()    
