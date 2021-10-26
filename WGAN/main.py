import argparse
import random
import torch
import util
import model
import classifier
import torch.backends.cudnn as cudnn
import torch.nn as nn
import numpy as np
import final_classifier
import torch.optim as optim
from torch.autograd import Variable
import torch.autograd as autograd
import time

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='SUN', help='AWA')
parser.add_argument('--syn_num', type=int, default=500, help='number features to generate per class')
parser.add_argument('--nepoch', type=int, default=2000, help='number of epochs to train for')
parser.add_argument('--gzsl',action='store_true', default=True, help='enable generalized zero-shot learning')
parser.add_argument('--attSize', type=int, default=102, help='size of semantic features')
parser.add_argument('--latent_dim', type=int, default=1024, help='size of latent features')
parser.add_argument('--nz', type=int, default=102, help='size of the latent z vector')
parser.add_argument('--lambda1', type=float, default=10, help='gradient penalty regularizer, following WGAN-GP')
parser.add_argument('--cls_weight', type=float, default=0.02, help='weight of the classification loss')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate to train GANs ')
parser.add_argument('--classifier_lr', type=float, default=0.001, help='learning rate to train softmax classifier')
parser.add_argument('--nclass_all', type=int, default=717, help='number of all classes')
parser.add_argument('--nclass_seen', type=int, default=645, help='number of seen classes')
parser.add_argument('--final_classifier', default='softmax', help='the classifier for final classification. softmax or knn')
parser.add_argument('--REG_W_LAMBDA',type=float,default=0.0001)

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

#init random seeds for every package
if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

#init torch settings
cudnn.benchmark = True
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# load data
data = util.DATA_LOADER(opt)
print("# of training samples: ", data.ntrain)


# initialize networks' structure
netG = model.MLP_G(opt)
netD = model.MLP_CRITIC(opt)
pre_G = model.PRE_G(opt)
BN = model.ClassStandardization(opt.resSize)

#init parameters and loss function
input_res = torch.FloatTensor(opt.batch_size, opt.resSize)
input_att = torch.FloatTensor(opt.batch_size, opt.attSize)
noise = torch.FloatTensor(opt.batch_size, opt.nz)
input_label = torch.LongTensor(opt.batch_size)
ones = torch.eye(opt.batch_size)
zeros = torch.zeros(size=(opt.batch_size, opt.batch_size))

cls_criterion = nn.NLLLoss()
loss_fn=torch.nn.MSELoss()

best_H=0
best_unseen = 0
consumed_time = []

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr,betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr,betas=(opt.beta1, 0.999))

if opt.cuda:
    netD.cuda()
    netG.cuda()
    pre_G.cuda()

    input_res = input_res.cuda()
    noise, input_att = noise.cuda(), input_att.cuda()
    cls_criterion.cuda()
    loss_fn.cuda()
    input_label = input_label.cuda()
    ones = ones.cuda()
    zeros = zeros.cuda()

#load pre_G setting
pre_G.load_state_dict(torch.load('pre_G/SUN/pre_netG_unseen0.19861111044883728_seen0.29612404108047485_H0.237757608294487.pkl'))

def sample():
    batch_feature, batch_label, batch_att = data.next_batch(opt.batch_size)
    input_res.copy_(batch_feature)
    input_att.copy_(batch_att)
    input_label.copy_(util.map_label(batch_label, data.seenclasses))

def sample_unseen(batch_size):
    random_indexs = np.random.randint(low=0,high=data.unseenclasses.shape[0],size=batch_size)
    random_unseen_attrs = data.attribute_unseen[random_indexs]
    return random_unseen_attrs.cuda()

def generate_syn_feature(netG, classes, attribute, num):
    nclass = classes.size(0)
    syn_feature = torch.FloatTensor(nclass*num, opt.resSize)
    syn_label = torch.LongTensor(nclass*num)
    syn_att = torch.FloatTensor(num, opt.attSize)
    syn_noise = torch.FloatTensor(num, opt.nz)
    if opt.cuda:
        syn_att = syn_att.cuda()
        syn_noise = syn_noise.cuda()
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

def compute_per_class_acc_gzsl(test_label, predicted_label, target_classes):
    acc_per_class = 0
    for i in target_classes:
        idx = (test_label == i)
        if torch.sum(idx)==0:
            acc_per_class +=0
        else:
            acc_per_class += float(torch.sum(test_label[idx] == predicted_label[idx])) / float(torch.sum(idx))
    acc_per_class /= float(target_classes.size(0))
    return acc_per_class

def calc_gradient_penalty(netD, real_data, fake_data, input_att):
    #print real_data.size()
    alpha = torch.rand(opt.batch_size, 1)
    alpha = alpha.expand(real_data.size())
    if opt.cuda:
        alpha = alpha.cuda()
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    if opt.cuda:
        interpolates = interpolates.cuda()

    interpolates = Variable(interpolates, requires_grad=True)
    disc_interpolates,_= netD(interpolates, Variable(input_att))
    ones = torch.ones(disc_interpolates.size())
    if opt.cuda:
        ones = ones.cuda()
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=ones,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * opt.lambda1
    return gradient_penalty

# train a classifier on seen classes, obtain \theta of Equation (4)
pretrain_cls = classifier.CLASSIFIER(data.train_feature, util.map_label(data.train_label, data.seenclasses), data.seenclasses.size(0), opt.resSize, opt.cuda, 0.001, 0.5, 50, 100)

for p in pretrain_cls.model.parameters(): # set requires_grad to False
    p.requires_grad = False

for epoch in range(opt.nepoch):
    start_time = time.time()
    for i in range(0, data.ntrain, opt.batch_size):
        for p in netD.parameters(): # reset requires_grad
            p.requires_grad = True # they are set to False below in netG update

        for iter_d in range(opt.critic_iter):
            sample()
            netD.zero_grad()

            input_resv = Variable(input_res)
            input_attv = Variable(input_att)

            criticD_real,pred_real = netD(input_resv, input_attv)
            criticD_real = criticD_real.mean()

            # train with fakeG
            noise.normal_(0, 1)
            noisev = Variable(noise)
            latent_fake = pre_G(noisev,input_attv)

            batch_mean, batch_var = BN(input_resv.detach().cpu())
            # batch_mean = batch_mean.unsqueeze(0).repeat(input_resv.shape[0], 1)
            # batch_var = batch_var.unsqueeze(0).repeat(input_resv.shape[0], 1)

            fake = netG(noisev, input_attv, latent_fake, batch_mean, batch_var)
            criticD_fake, pred_fake = netD(fake.detach(), input_attv)
            criticD_fake = criticD_fake.mean()

            noise.normal_(0, 1)
            unseen_atts = sample_unseen(opt.batch_size)
            latent_unseen_features = pre_G(noisev, unseen_atts)
            generated_unseen_features = netG(noisev, unseen_atts,latent_unseen_features, batch_mean, batch_var)

            ortho_same_loss = torch.sum(input_resv.mm(fake.t()) - ones)
            ortho_diff_loss = torch.sum(input_resv.mm(generated_unseen_features.t()) - zeros)
            ortho_diff_loss2 = torch.sum(fake.mm(generated_unseen_features.t()) - zeros)
            ortho_loss = ortho_same_loss + ortho_diff_loss + ortho_diff_loss2

            # gradient penalty
            gradient_penalty = calc_gradient_penalty(netD, input_resv, fake.data,input_attv)

            Wasserstein_D = criticD_real - criticD_fake
            D_cost = criticD_fake - criticD_real + gradient_penalty
            D_cost.backward()
            optimizerD.step()

        ############################
        # (2) Update G network: optimize WGAN-GP objective, Equation (2)
        ###########################
        for p in netD.parameters(): # reset requires_grad
            p.requires_grad = False # avoid computation

        netG.zero_grad()
        input_resv = Variable(input_res)
        input_attv = Variable(input_att)

        noise.normal_(0, 1)
        noisev = Variable(noise)
        latent_fake = pre_G(noise,input_attv)
        fake = netG(noisev, input_attv, latent_fake, batch_mean, batch_var)
        criticG_fake, pred_fake=netD(fake, input_attv, train_G=True)
        criticG_fake = criticG_fake.mean()
        G_cost = -criticG_fake
        c_errG_fake = cls_criterion(pretrain_cls.model(fake), input_label)

        reg_loss = Variable(torch.Tensor([0.0])).cuda()
        if opt.REG_W_LAMBDA != 0:
            for name, p in netG.named_parameters():
                if 'weight' in name:
                    reg_loss += p.pow(2).sum()
            reg_loss.mul_(opt.REG_W_LAMBDA)
        errG = G_cost + opt.cls_weight * c_errG_fake + reg_loss
        errG.backward()
        optimizerG.step()

    if opt.lr_dec:
        if (epoch + 1) % opt.lr_dec_ep == 0:
            for param_group in optimizerD.param_groups:
                param_group['lr'] = param_group['lr'] * opt.lr_dec_rate
            for param_group in optimizerG.param_groups:
                param_group['lr'] = param_group['lr'] * opt.lr_dec_rate

    end_time = time.time()
    consumed_time.append(end_time - start_time)
    print('[%d/%d] Loss_D: %.4f Loss_G: %.4f, Wasserstein_dist: %.4f, c_errG_fake:%.4f, Orl_loss:%.4f' % (
    epoch, opt.nepoch, D_cost.item(), G_cost.item(), Wasserstein_D.item(), c_errG_fake.item(), ortho_loss.item()))

    consumed_time_numpy = np.array(consumed_time)
    print('Average epoch time:', consumed_time_numpy.mean())
    print('Total epoch time:', consumed_time_numpy.sum())

    # evaluate the model, set G to evaluation mode
    netG.eval()
    netD.eval()

    # Generalized zero-shot learning
    syn_unseen_feature, syn_unseen_label = generate_syn_feature(netG, data.unseenclasses, data.attribute, opt.syn_num)
    train_X = torch.cat((data.train_feature, syn_unseen_feature), 0)
    train_Y = torch.cat((data.train_label, syn_unseen_label), 0)
    nclass = opt.nclass_all

    if opt.gzsl == True:
        cls = final_classifier.CLASSIFIER(train_X, train_Y, data, nclass, True, 0.001, 0.5, 50, 2 * opt.syn_num,True)
        print('unseen=%.4f, seen=%.4f, h=%.4f' % (cls.acc_unseen, cls.acc_seen, cls.H))
        if cls.H > best_H:
            best_H = cls.H
            # sio.savemat('syn_unseen.mat',{'unseen_feats': syn_unseen_feature.numpy(), 'unseen_labels': syn_unseen_label.numpy()})
            torch.save(netG.state_dict(),'./generator/seen{0}_unseen{1}_H{2}.pkl'.format(cls.acc_seen, cls.acc_unseen, cls.H))
            # torch.save(netD.state_dict(),'./discriminator/seen{0}_unseen{1}_H{2}.pkl'.format(cls.acc_seen, cls.acc_unseen, cls.H))
            print('best models saved!')
    else:
        syn_feature, syn_label = generate_syn_feature(netG, data.unseenclasses, data.attribute, opt.syn_num)
        cls = final_classifier.CLASSIFIER(syn_feature, util.map_label(syn_label, data.unseenclasses), data,data.unseenclasses.size(0), opt.cuda, opt.classifier_lr, 0.5, 50, 2 * opt.syn_num, False, epoch)
        if cls.acc > best_unseen:
            best_unseen = cls.acc
            print('best unseen acc is:', cls.acc)

    netG.train()
    netD.train()
