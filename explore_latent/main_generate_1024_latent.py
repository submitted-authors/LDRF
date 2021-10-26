import argparse
import util as util
import random
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from torch import nn
from models import MLP_G,MLP_D,Mapping
from torch.autograd import Variable
from pretrain_classifier import pretrain_classifier as pre
import classifier
from scipy import io as sio
import time

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='AWA2', help='AWA,AWA2,CUB,SUN,APY')
parser.add_argument('--dataroot', default='./dataset', help='path to data')
parser.add_argument('--image_embedding', default='res101')
parser.add_argument('--class_embedding', default='att')
parser.add_argument('--att_dim',default=85,help='att dim of dataset')
parser.add_argument('--noise_dim',default=85,help='noise dim setting')
parser.add_argument('--hidden_dim',default=4096,help='the dim of hidden layer for generator')
parser.add_argument('--latent_dim',default=1024,help='the dim of latent space')
parser.add_argument('--visual_dim',default=2048,help='the dim of visual features')
parser.add_argument('--nclass',default=50,help='the number of classes for dataset')
parser.add_argument('--reseen_classes',default=30,help='the number of resplit seen classes for dataset')
parser.add_argument('--reunseen_classes',default=10,help='the number of resplit unseen classes for dataset')
parser.add_argument('--batch_size',default=64,help='batch size for training')
parser.add_argument('--lr',default=0.0001,help='learning rate for training')
parser.add_argument('--cls_batch',default=500,help='the number of generated samples')
parser.add_argument('--syn_seen_num',default=2000,help='the number of generated seen samples')
parser.add_argument('--syn_unseen_num',default=2000,help='the number of generated seen samples')

parser.add_argument('--nepoch',default=1000,help='the number of training epoches')
parser.add_argument('--dis_epoch',default=1,help='the number of iterring discriminator network')
parser.add_argument('--gen_epoch',default=1,help='the number of iterring generator network')
opt = parser.parse_args()

#determine which GPU to run program
torch.cuda.set_device(0)

#init settings
seed = random.randint(0,10000)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
cudnn.benckmark =True

#load dataset
data = util.DATA_LOADER(opt)

#init important parameters
input_res = torch.FloatTensor(opt.batch_size,opt.visual_dim).cuda()
input_att = torch.FloatTensor(opt.batch_size,opt.att_dim).cuda()
unseen_samples = torch.FloatTensor(opt.batch_size,opt.att_dim).cuda()
input_label = torch.LongTensor(opt.batch_size).cuda()
original_label = torch.LongTensor(opt.batch_size).cuda()
noise = torch.FloatTensor(opt.batch_size,opt.noise_dim).cuda()
ones = torch.eye(opt.batch_size).cuda()
zeros = torch.zeros(size=(opt.batch_size,opt.batch_size)).cuda()

#init networks' structure
netG = MLP_G(opt).cuda()
netD = MLP_D(opt).cuda()
mapping = Mapping(opt).cuda()

#init networks' loss
cls_criterion = nn.NLLLoss().cuda()
mapping_criterion = nn.MSELoss(reduction='mean').cuda()

#init neteorks' optimizer
optimizerD = torch.optim.Adam(netD.parameters(),lr=opt.lr,betas=(0.5,0.999))
optimizerG = torch.optim.Adam(netG.parameters(),lr=opt.lr,betas=(0.5,0.999))
optimizer_mapping = torch.optim.Adam(mapping.parameters(),lr=opt.lr,betas=(0.5,0.999))

#init records' results
best_H = 0
ORL_LOSS = []
best_unseen = 0
consumed_time = []

#init useful functions
def sample():
    batch_features,batch_labels,batch_atts = data.next_batch(opt.batch_size)
    input_res.copy_(batch_features)
    input_att.copy_(batch_atts)
    original_label.copy_(batch_labels)
    input_label.copy_(util.map_label(batch_labels,data.seenclasses))#convert labels to 0-39

def sample_unseen(batch_size):
    random_indexs = np.random.randint(low=0,high=data.unseenclasses.shape[0],size=batch_size)
    random_unseen_attrs = data.attribute_unseen[random_indexs]
    return random_unseen_attrs.cuda()

def cal_gradient_penalty(netD,real_data,fake_data):
    alpha = torch.rand(opt.batch_size,1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda()

    interpolates = alpha*real_data+((1-alpha)*fake_data)
    interpolates = interpolates.cuda()
    interpolates = Variable(interpolates, requires_grad=True)

    dis,pred = netD(interpolates)
    grad_ones = torch.ones(dis.size()).cuda()
    gradients = torch.autograd.grad(outputs=dis, inputs=interpolates,
                              grad_outputs=grad_ones,create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()*10
    return gradient_penalty

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
            output = netG(syn_noise, syn_att)
            syn_feature.narrow(0, i * num, num).copy_(output.data.cpu())
            syn_label.narrow(0, i * num, num).fill_(iclass)
    return syn_feature, syn_label

#training the latent GAN
for epoch in range(opt.nepoch):
    # train a pre-train classifier for seen classes,and set requires_grad to False
    pretrain_x = data.train_feature.cuda()
    pretrain_x = mapping(pretrain_x)
    pretrain_cls = pre(pretrain_x, util.map_label(data.train_label, data.seenclasses),
                       data.seenclasses.size(0), opt.latent_dim, True, 0.001, 0.5, 100, 100)
    for p in pretrain_cls.model.parameters():
        p.requires_grad = False

    start_time = time.time()
    for i in range(0,data.ntrain,opt.batch_size):
        #when iter GAN,set mapping and netG to False
        for p in mapping.parameters():
            p.requires_grad = True

        for p in netG.parameters():
            p.requires_grad = True

        #iter generator of GANs
        for iter_g in range(opt.gen_epoch):
            sample()
            netG.zero_grad()
            mapping.zero_grad()

            seen_features = Variable(input_res)
            seen_atts = Variable(input_att)

            real_mapping = mapping(seen_features)  # [512,1024]

            noise.normal_(0, 1)
            noisev = Variable(noise)

            unseen_atts = sample_unseen(opt.batch_size)
            generated_seen_mapping = netG(noisev, seen_atts)  # [512,1024]
            generated_unseen_mapping = netG(noisev, unseen_atts)  # [512,1024]

            ortho_same_loss = torch.sum(real_mapping.mm(generated_seen_mapping.t()) - ones)
            ortho_diff_loss = torch.sum(real_mapping.mm(generated_unseen_mapping.t()) - zeros)
            ortho_diff_loss2 = torch.sum(generated_seen_mapping.mm(generated_unseen_mapping.t()) - zeros)
            ortho_loss = ortho_same_loss + ortho_diff_loss + ortho_diff_loss2

            mapping_loss = mapping_criterion(real_mapping, generated_seen_mapping)

            noise.normal_(0,1)
            noisev = Variable(noise)

            fake_mapping = netG(noisev,seen_atts)
            fake_dis,fake_pred = netD(fake_mapping)
            criticG_fake = fake_dis.mean()

            cls_fake = cls_criterion(pretrain_cls.model(fake_mapping),input_label)
            cls_real = cls_criterion(pretrain_cls.model(real_mapping),input_label)
            G_loss = -criticG_fake + 0.2 * cls_fake + 0.1 * cls_real + 0.1 * mapping_loss + ortho_loss * 1e-8
            G_loss.backward()
            optimizer_mapping.step()
            optimizerG.step()

        #iter GAN networks and pretrain classifier
        for p in mapping.parameters():
            p.requires_grad = False

        for p in netG.parameters():
            p.requires_grad = False


        #iter discriminator of GANs
        for p in netD.parameters():
            p.requires_grad = True

        for iter_d in range(opt.dis_epoch):
            sample()
            netD.zero_grad()

            seen_features = Variable(input_res)
            seen_atts = Variable(input_att)

            #train for true samples
            real_mapping = mapping(seen_features)
            real_dis, real_pred = netD(real_mapping)
            criticD_real = real_dis.mean()

            #train for fake samples
            noise.normal_(0,1)
            noisev = Variable(noise)

            fake_mapping = netG(noisev,seen_atts)
            fake_dis,fake_pred = netD(fake_mapping)
            criticD_fake = fake_dis.mean()

            #cal gradient penalty
            gradient_penalty = cal_gradient_penalty(netD,real_mapping,fake_mapping)

            Wasserstein_loss = criticD_fake - criticD_real + gradient_penalty
            Wasserstein_loss.backward()
            optimizerD.step()

        for p in netD.parameters():
            p.requires_grad = False
    end_time = time.time()
    consumed_time.append(end_time-start_time)
    for p in pretrain_cls.model.parameters():
        p.requires_grad = True

    print('[%d/%d] Ortho_Loss:%.4f, mappingLoss:%.4f, Wasserstein_Loss:%.4f, G_Loss:%.4f'% (epoch, opt.nepoch, ortho_loss.item(), mapping_loss.item(), Wasserstein_loss.item(), G_loss.item()))
    ORL_LOSS.append(ortho_loss.item())

    consumed_time_numpy = np.array(consumed_time)
    print('Average epoch time:',consumed_time_numpy.mean())
    print('Total epoch time:',consumed_time_numpy.sum())
    # evaluate the model
    mapping.eval()
    netG.eval()
    syn_unseen_mapping, syn_unseen_label = generate_mapping(netG, data.unseenclasses, data.attribute, opt.syn_unseen_num)  # [n,1024] and [n]
    syn_unseen_mapping = syn_unseen_mapping.cuda()
    syn_unseen_label = syn_unseen_label.cuda()

    syn_seen_mapping, syn_seen_label = generate_mapping(netG, data.seenclasses, data.attribute, opt.syn_seen_num)
    syn_seen_mapping = syn_seen_mapping.cuda()
    syn_seen_label = syn_seen_label.cuda()

    train_feature = data.train_feature.cuda()
    train_mapping = mapping(train_feature)  # [19832,1024]
    train_label = data.train_label.cuda()

    # train_X = torch.cat((train_mapping, syn_unseen_mapping), 0)
    # train_Y = torch.cat((train_label, syn_unseen_label), 0)

    train_X = torch.cat((syn_seen_mapping, syn_unseen_mapping), 0)
    train_Y = torch.cat((syn_seen_label, syn_unseen_label), 0)

    nclass = opt.nclass

    cls = classifier.CLASSIFIER(mapping, train_X, train_Y, data, nclass, True, 0.001, 0.5, 50, 2 * opt.cls_batch,True)
    print('unseen=%.4f, seen=%.4f, h=%.4f' % (cls.acc_unseen, cls.acc_seen, cls.H))
    PATH = './1024_latent'
    if cls.H > best_H:
        best_H = cls.H
        torch.save(netG.state_dict(),PATH + '/pre_netG_unseen{0}_seen{1}_H{2}.pkl'.format(cls.acc_unseen, cls.acc_seen, cls.H))
        print('best model saved!!!')

    # cls = classifier.CLASSIFIER(mapping,syn_unseen_mapping, util.map_label(syn_unseen_label, data.unseenclasses), data,
    #                              data.unseenclasses.size(0), True, 0.001, 0.5, 50, 2 * opt.cls_batch,
    #                              False, epoch)
    # PATH = './saved_model'
    # if cls.acc > best_unseen:
    #     best_unseen = cls.acc
    #     torch.save(netG.state_dict(), PATH + '/pre_netG_acc{0}.pkl'.format(cls.acc))
    #     print('best unseen acc is:', cls.acc)

    mapping.train()
    netG.train()

#save orth loss as mat
loss = np.array(ORL_LOSS)
sio.savemat('AWA2.mat', {'orl_loss':loss})