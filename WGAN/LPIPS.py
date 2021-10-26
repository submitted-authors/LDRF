import lpips
import util
import model
import numpy as np
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
import model

lpips_loss_alex= lpips.LPIPS('alex')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='APY', help='AWA')
parser.add_argument('--syn_num', type=int, default=10, help='number features to generate per class')
parser.add_argument('--nepoch', type=int, default=2000, help='number of epochs to train for')
parser.add_argument('--gzsl',action='store_true', default=True, help='enable generalized zero-shot learning')
parser.add_argument('--attSize', type=int, default=64, help='size of semantic features')
parser.add_argument('--latent_dim', type=int, default=1024, help='size of latent features')
parser.add_argument('--nz', type=int, default=64, help='size of the latent z vector')
parser.add_argument('--lambda1', type=float, default=10, help='gradient penalty regularizer, following WGAN-GP')
parser.add_argument('--cls_weight', type=float, default=0.02, help='weight of the classification loss')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate to train GANs ')
parser.add_argument('--classifier_lr', type=float, default=0.001, help='learning rate to train softmax classifier')
parser.add_argument('--nclass_all', type=int, default=32, help='number of all classes')
parser.add_argument('--nclass_seen', type=int, default=20, help='number of seen classes')
parser.add_argument('--final_classifier', default='softmax', help='the classifier for final classification. softmax or knn')
parser.add_argument('--REG_W_LAMBDA',type=float,default=0.0004)

parser.add_argument('--dataroot', default='./dataset', help='path to dataset')
parser.add_argument('--matdataset', default=True, help='Data in matlab format')
parser.add_argument('--image_embedding', default='res101')
parser.add_argument('--class_embedding', default='att')
parser.add_argument('--preprocessing', action='store_true',  default=True, help='enbale MinMaxScaler on visual features')
parser.add_argument('--standardization', action='store_true', default=False)
parser.add_argument('--validation', action='store_true', default=False, help='enable cross validation mode')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batch_size', type=int, default=512, help='input batch size')
parser.add_argument('--resSize', type=int, default=2048, help='size of visual features')
parser.add_argument('--ndh', type=int, default=4096, help='size of the hidden units in discriminator')
parser.add_argument('--ngh', type=int, default=4096, help='size of the hidden units in generator')
parser.add_argument('--critic_iter', type=int, default=5, help='critic iteration, following WGAN-GP')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', default=True, help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--manualSeed', type=int, default=None,help='manual seed')#3483
parser.add_argument('--lr_dec', action='store_true', default=True, help='enable lr decay or not')
parser.add_argument('--lr_dec_ep', type=int, default=100, help='lr decay for every 100 epoch')
parser.add_argument('--lr_dec_rate', type=float, default=0.95, help='lr decay rate')
opt = parser.parse_args()

def generate_syn_feature(netG, classes, attribute, num):
    nclass = classes.size(0)
    syn_feature = torch.FloatTensor(nclass*num, opt.resSize)
    syn_label = torch.LongTensor(nclass*num)
    syn_att = torch.FloatTensor(num, opt.attSize)
    syn_noise = torch.FloatTensor(num, opt.nz)
    if opt.cuda:
        syn_att = syn_att
        syn_noise = syn_noise
    with torch.no_grad():
        for i in range(nclass):
            iclass = classes[i]
            iclass_att = attribute[iclass]
            syn_att.copy_(iclass_att.repeat(num, 1))
            syn_noise.normal_(0, 1)
            latent_x = pre_G(syn_noise,syn_att)
            running_mean = Variable(BN.running_mean.cuda())
            running_var = Variable(BN.running_var.cuda())
            output = netG(syn_noise, syn_att, latent_x, running_mean, running_var)
            syn_feature.narrow(0, i*num, num).copy_(output.data.cpu())
            syn_label.narrow(0, i*num, num).fill_(iclass)
    return syn_feature, syn_label

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
        self.fc1 = nn.Linear(opt.nz + opt.attSize + opt.latent_dim, opt.ngh)
        # self.fc1 = nn.Linear(opt.nz + opt.attSize, opt.ngh)
        self.fc2 = nn.Linear(opt.ngh, opt.resSize)
        self.fc3 = nn.Linear(opt.resSize, opt.resSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)
        self.apply(weights_init)

    def forward(self, noise, att, latent_x, mean, var):
        h = torch.cat((att, noise,latent_x), 1)
        # h = torch.cat((att, noise), 1)
        h = self.lrelu(self.fc1(h))
        h = self.relu(self.fc2(h))
        return h


class PRE_G(nn.Module):
    def __init__(self, opt):
        super(PRE_G, self).__init__()
        self.fc1 = nn.Linear(opt.nz+opt.attSize,opt.ngh)
        self.fc2 = nn.Linear(opt.ngh, opt.latent_dim)
        self.lrelu = nn.LeakyReLU(0.2,True)
        self.relu = nn.ReLU(True)
        self.apply(weights_init)

    def forward(self,noise,att):
        hidden = torch.cat((noise,att),dim=1)
        hidden = self.lrelu(self.fc1(hidden))
        latent_out = self.relu(self.fc2(hidden))
        return latent_out

data = util.DATA_LOADER(opt)
netG = MLP_G(opt)
pre_G = PRE_G(opt)
BN = model.ClassStandardization(opt.resSize)

netG.load_state_dict(torch.load('FID_AND_LPIPS/APY/seen0.6137548089027405_unseen0.3391527831554413_H0.43688738346099854.pkl'))
pre_G.load_state_dict(torch.load('pre_G/APY/pre_netG_unseen0.15690574049949646_seen0.4675293564796448_H0.23495809733867645.pkl'))
print('load successfully!')

print('begin generating unseen samples...')
tmpx, tmpy = generate_syn_feature(netG, data.unseenclasses, data.attribute, opt.syn_num)
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