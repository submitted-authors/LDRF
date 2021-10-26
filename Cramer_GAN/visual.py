import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from sklearn import manifold
from model import MLP_G,PRE_G
import argparse
import util
from scipy import io as sio
import matplotlib.pyplot as plt
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='AWA', help='dataset for zsl dataset')
parser.add_argument('--syn_num', type=int, default=1800, help='number features to generate per class')
parser.add_argument('--syn_unseen_num', type=int, default=1800, help='number features to generate unseen classes')
parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
parser.add_argument('--nepoch', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--resSize', type=int, default=2048, help='size of visual features')
parser.add_argument('--attSize', type=int, default=85, help='size of semantic features')
parser.add_argument('--nz', type=int, default=85, help='size of the latent z vector')
parser.add_argument('--latent_dim',type=int,default=1024)
parser.add_argument('--lambda1', type=float, default=2, help='gradient penalty regularizer, following WGAN-GP')
parser.add_argument('--cls_weight', type=float, default=0.01, help='weight of the classification loss')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate to train GANs ')
parser.add_argument('--classifier_lr', type=float, default=0.001, help='learning rate to train softmax classifier')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--nclass_all', type=int, default=50, help='number of all classes')
parser.add_argument('--proto_param1', type=float, default=1.0, help='proto param 1')
parser.add_argument('--REG_W_LAMBDA',type=float,default=0.0,help='regularization param')

parser.add_argument('--critic_iter', type=int, default=5, help='critic iteration, following WGAN-GP')
parser.add_argument('--ngh', type=int, default=4096, help='size of the hidden units in generator')
parser.add_argument('--ndh', type=int, default=4096, help='size of the hidden units in discriminator')
parser.add_argument('--netG_name', default='MLP_G')
parser.add_argument('--netD_name', default='MLP_CRITIC')
parser.add_argument('--save_every', type=int, default=100)
parser.add_argument('--print_every', type=int, default=1)
parser.add_argument('--val_every', type=int, default=1)
parser.add_argument('--start_epoch', type=int, default=0)

parser.add_argument('--dataroot', default='./data', help='path to dataset')
parser.add_argument('--matdataset', default=True, help='Data in matlab format')
parser.add_argument('--image_embedding', default='res101')
parser.add_argument('--class_embedding', default='att')
parser.add_argument('--gzsl', action='store_true', default=True, help='enable generalized zero-shot learning')
parser.add_argument('--preprocessing', action='store_true', default=True,help='enbale MinMaxScaler on visual features')
parser.add_argument('--standardization', action='store_true', default=False)
parser.add_argument('--validation', action='store_true', default=False, help='enable cross validation mode')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--cuda', action='store_true', default=True, help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--pretrain_classifier', default='', help="path to pretrain classifier (to continue training)")
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='./checkpoint/', help='folder to output data and model checkpoints')
parser.add_argument('--outname', help='folder to output data and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')

#param init
opt = parser.parse_args()
PATH='generator/SYN2000/SEED589-0.0004/seen0.7270665168762207_unseen0.58918297290802_H0.6509027481079102.pkl'
netG=MLP_G(opt).cuda()
netG.load_state_dict(torch.load(PATH))
print(netG)

pre_G = PRE_G(opt).cuda()
pre_G.load_state_dict(torch.load('pre_G/pre_netG_unseen0.3371211886405945_seen0.7699975371360779_H0.46893343329429626.pkl'))

def generate_syn_feature(netG,pre_G,classes, attribute, num):
    nclass = classes.size(0)
    syn_feature = torch.FloatTensor(nclass * num, opt.resSize)
    syn_label = torch.LongTensor(nclass * num)
    syn_att = torch.FloatTensor(num, opt.attSize)
    syn_noise = torch.FloatTensor(num, opt.nz)
    if opt.cuda:
        syn_att = syn_att.cuda()
        syn_noise = syn_noise.cuda()

    for i in range(nclass):
        iclass = classes[i]
        iclass_att = attribute[iclass]
        syn_att.copy_(iclass_att.repeat(num, 1))
        syn_noise.normal_(0, 1)
        latent = pre_G(Variable(syn_noise), Variable(syn_att))
        output = netG(Variable(latent),Variable(syn_noise), Variable(syn_att))
        syn_feature.narrow(0, i * num, num).copy_(output.data.cpu())
        syn_label.narrow(0, i * num, num).fill_(iclass)
    return syn_feature, syn_label

data = util.DATA_LOADER(opt)
syn_feature, syn_label = generate_syn_feature(netG,pre_G, data.unseenclasses, data.attribute, opt.syn_num)
print(syn_feature.shape)
print(syn_label.shape)
sio.savemat('./label.mat',{'features':syn_feature.numpy(),'labels':syn_label.numpy()})
# def visualization(feature, label, save_dir, nameStr):
#     '''t-SNE visualization for visual features.'''
#     assert feature.shape[0] == label.shape[0]
#     X = feature
#     labels = label
#     sne = manifold.TSNE(n_components=2, init='pca', random_state=501)
#     Y=sne.fit_transform(X)
#     plt.scatter(Y[:, 0], Y[:, 1], 20, labels)
#     save_path = os.path.join(save_dir, nameStr + '.png')
#     plt.savefig(save_path)
#     print('visualization results are saved done in %s!' % save_dir)
# visualization(syn_feature.numpy(),syn_label.numpy(),'./result1','awa1')
