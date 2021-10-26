import lpips
from torch import nn
import torch
import utils_all as util
import numpy as np
lpips_loss_alex= lpips.LPIPS('alex')
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='APY', help='AWA,AWA2,CUB,SUN,APY')
parser.add_argument('--dataroot', default='./dataset', help='path to data')
parser.add_argument('--image_embedding', default='res101')
parser.add_argument('--class_embedding', default='att')
parser.add_argument('--att_dim',default=64,help='att dim of dataset')
parser.add_argument('--noise_dim',default=64,help='noise dim setting')
parser.add_argument('--hidden_dim',default=4096,help='the dim of hidden layer for generator')
parser.add_argument('--latent_dim',default=2048,help='the dim of latent space')
parser.add_argument('--visual_dim',default=2048,help='the dim of visual features')
parser.add_argument('--nclass',default=32,help='the number of classes for dataset')
parser.add_argument('--reseen_classes',default=12,help='the number of resplit seen classes for dataset')
parser.add_argument('--reunseen_classes',default=8,help='the number of resplit unseen classes for dataset')
parser.add_argument('--batch_size',default=64,help='batch size for training')
parser.add_argument('--lr',default=0.00001,help='learning rate for training')
parser.add_argument('--cls_batch',default=500,help='the number of generated samples')
parser.add_argument('--syn_seen_num',default=10,help='the number of generated seen samples')
parser.add_argument('--syn_unseen_num',default=10,help='the number of generated seen samples')

parser.add_argument('--nepoch',default=1000,help='the number of training epoches')
parser.add_argument('--dis_epoch',default=1,help='the number of iterring discriminator network')
parser.add_argument('--gen_epoch',default=1,help='the number of iterring generator network')
opt = parser.parse_args()

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class MLP_G(nn.Module):
    def __init__(self, opt):
        super(MLP_G, self).__init__()
        self.fc1 = nn.Linear(opt.noise_dim+opt.att_dim,opt.hidden_dim)
        self.fc2 = nn.Linear(opt.hidden_dim, opt.latent_dim)
        self.lrelu = nn.LeakyReLU(0.2,True)
        self.relu = nn.ReLU(True)
        self.apply(weights_init)

    def forward(self,noise,att):
        hidden = torch.cat((noise,att),dim=1).cuda()
        hidden = self.lrelu(self.fc1(hidden))
        latent_out = self.relu(self.fc2(hidden))
        return latent_out

def generate_mapping(netG,classes,attribute,num):
    nclass = classes.size(0)
    syn_feature = torch.FloatTensor(nclass * num, opt.latent_dim)
    syn_label = torch.LongTensor(nclass * num)
    syn_att = torch.FloatTensor(num, opt.att_dim).cuda()
    syn_noise = torch.FloatTensor(num, opt.noise_dim).cuda()
    with torch.no_grad():
        for i in range(nclass):
            iclass = classes[i]
            iclass_att = attribute[iclass]
            syn_att.copy_(iclass_att.repeat(num, 1))
            syn_noise.normal_(0, 1)
            syn_noise = syn_noise.cuda()
            syn_att = syn_att.cuda()
            output = netG(syn_noise, syn_att)
            syn_feature.narrow(0, i * num, num).copy_(output.data.cpu())
            syn_label.narrow(0, i * num, num).fill_(iclass)
    return syn_feature, syn_label

data = util.DATA_LOADER(opt)
pre_G = MLP_G(opt).cuda()
pre_G.load_state_dict(torch.load('saved_model/FID_AND_LPIPS/APY/pre_netG_unseen0.1612538993358612_seen0.563572347164154_H0.25075870752334595.pkl'))

print('load successfully!')

print('begin generating unseen samples...')
tmpx, tmpy = generate_mapping(pre_G, data.unseenclasses, data.attribute, opt.syn_unseen_num)
print(tmpx.shape)
print(tmpy.shape)

def change(x):
    x = x.view(2,32,32)
    x1 = x[0,:,:]
    x2 = x[0,:,:]
    x3 = (x1+x2)/2
    x = torch.stack((x1,x3,x2))
    return x
def calLPIPS(x,y,unseen_class):
    x = x.numpy()
    y = y.numpy()
    unseen_class = unseen_class.numpy()

    total_loss = 0
    CLUS = np.zeros((unseen_class.shape[0],2048))
    for idx in range(unseen_class.shape[0]):
        tmpx = np.zeros((1, 2048))
        for i in range(y.shape[0]):
            if(y[i]==unseen_class[idx]):
                tmpx += x[i,:]
        CLUS[idx,:] = tmpx

    for idx in range(unseen_class.shape[0]):
        loss = 0
        number = 0
        for i in range(y.shape[0]):
            if(y[i]==unseen_class[idx]):
                number +=1
                im1,im2 = change(torch.tensor(x[i,:])),change(torch.tensor(CLUS[idx,:]))
                im1 = im1.to(torch.float32)
                im2 = im2.to(torch.float32)
                loss_ = lpips_loss_alex(im1,im2)
                loss += loss_
        total_loss += loss/number

    total_loss /= unseen_class.shape[0]
    return total_loss

# print(data.test_unseen_label)

max_LPIPS = 0
# print(torch.unique(data.test_unseen_label).shape)
LPIPS = calLPIPS(tmpx, tmpy, torch.unique(data.test_unseen_label))
max_LPIPS = max_LPIPS if max_LPIPS > LPIPS else LPIPS
print(max_LPIPS)