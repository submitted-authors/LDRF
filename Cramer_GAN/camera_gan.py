from __future__ import print_function
import argparse
import random
import torch
import torch.nn as nn
import scipy.io as sio
import torch.autograd as autograd
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import util
import classifier
import classifier2
import classifier_latent
import model
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import time

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='APY', help='dataset for zsl dataset')
parser.add_argument('--syn_num', type=int, default=2200, help='number features to generate per class')
parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
parser.add_argument('--nepoch', type=int, default=1000, help='number of epochs to train for')
parser.add_argument('--resSize', type=int, default=2048, help='size of visual features')
parser.add_argument('--attSize', type=int, default=64, help='size of semantic features')
parser.add_argument('--nz', type=int, default=64, help='size of the latent z vector')
parser.add_argument('--latent_dim',type=int,default=1024)
parser.add_argument('--lambda1', type=float, default=2, help='gradient penalty regularizer, following WGAN-GP')
parser.add_argument('--cls_weight', type=float, default=0.01, help='weight of the classification loss')
parser.add_argument('--lr', type=float, default=0.00001, help='learning rate to train GANs ')
parser.add_argument('--classifier_lr', type=float, default=0.001, help='learning rate to train softmax classifier')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--nclass_all', type=int, default=32, help='number of all classes')
parser.add_argument('--proto_param1', type=float, default=1.0, help='proto param 1')
parser.add_argument('--REG_W_LAMBDA',type=float,default=0.0004,help='regularization param')
parser.add_argument('--gzsl', action='store_true', default=True, help='enable generalized zero-shot learning')
parser.add_argument('--final_classifier',default='softmax',help='softmax or knn')

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
parser.add_argument('--preprocessing', action='store_true', default=True,help='enbale MinMaxScaler on visual features')
parser.add_argument('--standardization', action='store_true', default=False)
parser.add_argument('--validation', action='store_true', default=False, help='enable cross validation mode')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--cuda', action='store_true', default=True, help='enables cuda')
parser.add_argument('--ngpu', type=int, default=0, help='number of GPUs to use')
parser.add_argument('--pretrain_classifier', default='', help="path to pretrain classifier (to continue training)")
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='./checkpoint/', help='folder to output data and model checkpoints')
parser.add_argument('--outname', help='folder to output data and model checkpoints')
parser.add_argument('--manualSeed', type=int,default=None,help='manual seed')

#param init
opt = parser.parse_args()
torch.cuda.set_device(opt.ngpu)
print('Params: dataset={:s}, GZSL={:s}, cls_weight={:.4f}, proto_param1={:.4f}'.format(opt.dataset, str(opt.gzsl), opt.cls_weight,opt.proto_param1))
if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)
cudnn.benchmark = True
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# load data
data = util.DATA_LOADER(opt)
print("Training samples: ", data.ntrain)#19832

# initialize generator and discriminator
netG = model.MLP_G(opt)
pre_G = model.PRE_G(opt)
BN = model.ClassStandardization(2048)
# unseen_BN = model.ClassStandardization(1024)

if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

netD = model.MLP_D(opt)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

# classification loss, Equation (4) of the paper
cls_criterion = nn.NLLLoss()

input_res = torch.FloatTensor(opt.batch_size, opt.resSize)#[64,2048]
input_att = torch.FloatTensor(opt.batch_size, opt.attSize)#[64,85]
noise = torch.FloatTensor(opt.batch_size, opt.nz)#[64,85]
noise2 = torch.FloatTensor(opt.batch_size, opt.nz)#[64,85]

one = torch.FloatTensor([1])
mone = one * -1
input_label = torch.LongTensor(opt.batch_size)#[64,]

if opt.cuda:
    netD.cuda()
    netG.cuda()
    pre_G.cuda()
    BN.cuda()
    # unseen_BN.cuda()
    input_res = input_res.cuda()
    noise,noise2, input_att = noise.cuda(),noise2.cuda(), input_att.cuda()
    one = one.cuda()
    mone = mone.cuda()
    cls_criterion.cuda()
    input_label = input_label.cuda()
    ones = torch.eye(opt.batch_size).cuda()
    zeros = torch.zeros(size=(opt.batch_size, opt.batch_size)).cuda()

pre_G.load_state_dict(torch.load('pre_G/APY/pre_netG_unseen0.15690574049949646_seen0.4675293564796448_H0.23495809733867645.pkl'))
def sample():
    batch_feature, batch_label, batch_att = data.next_batch(opt.batch_size)#s label is normal label based 0
    input_res.copy_(batch_feature)
    input_att.copy_(batch_att)
    input_label.copy_(util.map_label(batch_label, data.seenclasses))#map normal label into 0-39

def sample_unseen(batch_size):
    random_indexs = np.random.randint(low=0,high=data.unseenclasses.shape[0],size=batch_size)
    random_unseen_attrs = data.attribute_unseen[random_indexs]
    return random_unseen_attrs.cuda()

def generate_syn_feature(netG, classes, attribute, num):
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
        latent_feature = pre_G(syn_noise,syn_att)
        output = netG(Variable(latent_feature), Variable(syn_noise), Variable(syn_att),BN.running_mean,BN.running_var)
        syn_feature.narrow(0, i * num, num).copy_(output.data.cpu())
        syn_label.narrow(0, i * num, num).fill_(iclass)
    return syn_feature, syn_label

def map_label(label, classes):
    mapped_label = torch.LongTensor(label.size())
    for i in range(classes.size(0)):
        mapped_label[label==classes[i]] = i
    return mapped_label

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
# optimizerD=optim.RMSprop(netG.parameters(),lr=opt.lr, alpha=0.9)
# optimizerG=optim.RMSprop(netD.parameters(),lr=opt.lr,alpha=0.9)
result_seen=[]
result_unseen=[]
result_H=[]
consumed_time = []

def calc_gradient_penalty(netD, real_data, fake_data, input_att):
    # print real_data.size()
    alpha = torch.rand(opt.batch_size, 1)
    alpha = alpha.expand(real_data.size())
    if opt.cuda:
        alpha = alpha.cuda()
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    if opt.cuda:
        interpolates = interpolates.cuda()
    interpolates = Variable(interpolates, requires_grad=True)
    disc_interpolates,_ = netD(interpolates, Variable(input_att))
    ones = torch.ones(disc_interpolates.size())
    if opt.cuda:
        ones = ones.cuda()
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=ones,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1)) ** 6).mean() * opt.lambda1
    return gradient_penalty

def Critic(netD, real, fake2,att):
    net_real,_ = netD(real,att)
    return torch.norm(net_real - netD(fake2,att)[0], p=2, dim=1) - \
           torch.norm(net_real, p =2,  dim=1)

def compute_per_class_acc_gzsl(predicted_label, test_label, target_classes):
    acc_per_class = 0
    for i in target_classes:
        idx = (test_label == i)
        if torch.sum(idx).float()==0:
            continue
        else:
            acc_per_class += torch.sum(test_label[idx] == predicted_label[idx]).float() / torch.sum(idx).float()
    acc_per_class /= target_classes.size(0)
    return acc_per_class

# train a classifier on seen classes, obtain \theta of Equation (4)
pretrain_cls = classifier.CLASSIFIER(_train_X=data.train_feature, _train_Y=util.map_label(data.train_label, data.seenclasses),
                                     _nclass=data.seenclasses.size(0), _input_dim=opt.resSize, _cuda=opt.cuda, _lr=0.001, _beta1=0.5, _nepoch=100, _batch_size=100,
                                     pretrain_classifer=opt.pretrain_classifier)

# freeze the classifier during the optimization
for p in pretrain_cls.model.parameters():  # set requires_grad to False
    p.requires_grad = False

best_H=0
best_H_0 = 0
best_unseen=0
loss_fn=nn.CrossEntropyLoss()
for epoch in range(opt.nepoch):
    start_time = time.time()
    for i in range(0, data.ntrain, opt.batch_size):
        for p in netD.parameters():
            p.requires_grad = True
        #optimize discriminator
        for iter_d in range(opt.critic_iter):#5
            sample()#samples for input_res,input_att,input_label
            netD.zero_grad()
            input_resv = Variable(input_res)#[64,2048]
            batch_mean, batch_var = BN(input_resv)
            input_attv = Variable(input_att)#[64,85]
            input_labelv=Variable(input_label)#[64]

            criticD_real,pred_real = netD(input_resv, input_attv)
            criticD_real_loss = criticD_real.mean()
            # criticD_real_loss.backward(mone,retain_graph=True)

            noise.normal_(0, 1)
            noise2.normal_(0,1)
            noisev = Variable(noise)#[64,85]
            noisev2=Variable(noise2)#[64,85]

            latent_feat1 = pre_G(noisev,input_attv)
            fake = netG(latent_feat1, noisev, input_attv, batch_mean, batch_var)
            latent_feat2 = pre_G(noisev2,input_attv)
            fake2=netG(latent_feat2,noisev2,input_attv, batch_mean, batch_var)

            criticD_fake,pred_fake = netD(fake.detach(), input_attv)
            criticD_fake2,pred_fake2 = netD(fake2.detach(), input_attv)

            criticD_fake_loss = criticD_fake.mean()
            # criticD_fake_loss.backward(one,retain_graph=True)

            noise.normal_(0, 1)
            noisev = Variable(noise)
            unseen_atts = sample_unseen(opt.batch_size)
            latent_unseen_features = pre_G(noisev, unseen_atts)
            # unseen_batch_mean, unseen_batch_mean = unseen_BN(latent_unseen_features)
            # generated_unseen_features = netG(latent_unseen_features,noisev, unseen_atts, unseen_batch_mean, unseen_batch_mean)
            generated_unseen_features = netG(latent_unseen_features, noisev, unseen_atts, batch_mean, batch_var)

            ortho_same_loss = torch.sum(input_resv.mm(fake.t()) - ones)
            ortho_same_loss2 = torch.sum(input_resv.mm(fake2.t()) - ones)
            ortho_diff_loss = torch.sum(input_resv.mm(generated_unseen_features.t()) - zeros)
            ortho_diff_loss2 = torch.sum(fake.mm(generated_unseen_features.t()) - zeros)
            ortho_diff_loss3 = torch.sum(fake2.mm(generated_unseen_features.t()) - zeros)
            ortho_loss = ortho_same_loss + ortho_same_loss2 + ortho_diff_loss + ortho_diff_loss2 + ortho_diff_loss3

            gen_loss = torch.mean(
                torch.norm(criticD_real - criticD_fake, p=2, dim=1)
                + torch.norm(criticD_real - criticD_fake2, p=2, dim=1)
                - torch.norm(criticD_fake - criticD_fake2, p=2, dim=1)
            )

            surrogate = torch.mean(Critic(netD, input_resv, fake2,input_attv) -Critic(netD, fake, fake2,input_attv))
            gradient_penalty = calc_gradient_penalty(netD, input_res, fake.data, input_att)
            # D_cost = criticD_fake_loss - criticD_real_loss + gradient_penalty-surrogate
            # D_cost.backward(retain_graph=True)

            disc_loss=-surrogate+gradient_penalty
            disc_loss.backward(retain_graph=True)
            # Wasserstein_D = criticD_real - criticD_fake
            # D_cost = criticD_fake_loss - criticD_real_loss + gradient_penalty
            # D_cost.backward(retain_graph=True)
            optimizerD.step()
        #stop optimize discriminator
        for p in netD.parameters():  # reset requires_grad
            p.requires_grad = False  # avoid computation

        #begin optimizing generator
        netG.zero_grad()
        input_resv = Variable(input_resv)
        input_attv = Variable(input_att)#[64,85]
        noise.normal_(0, 1)
        noisev = Variable(noise)#[64,85]
        latent_feat3 = pre_G(noisev,input_attv)
        fake = netG(latent_feat3, noisev, input_attv,batch_mean,batch_var)#[64,2048]
        criticG_fake,pred_fake = netD(fake, input_attv)
        criticG_fake = criticG_fake.mean()
        G_cost = -criticG_fake

        # classification loss
        c_errG = cls_criterion(pretrain_cls.model(fake), Variable(input_label))

        # ||W||_2 regularization
        reg_loss = Variable(torch.Tensor([0.0])).cuda()
        if opt.REG_W_LAMBDA != 0:
            for name, p in netG.named_parameters():
                if 'weight' in name:
                    reg_loss += p.pow(2).sum()
            reg_loss.mul_(opt.REG_W_LAMBDA)

        errG = G_cost + opt.cls_weight * c_errG + gen_loss * opt.proto_param1 + reg_loss
        errG.backward(retain_graph=True)
        optimizerG.step()
    end_time = time.time()
    consumed_time.append(end_time - start_time)
    print('EP[%d/%d]************************************************************************************' % (epoch, opt.nepoch))
    consumed_time_numpy = np.array(consumed_time)
    print('Average epoch time:', consumed_time_numpy.mean())
    print('Total epoch time:', consumed_time_numpy.sum())

    # evaluate the model, set G to evaluation mode
    netG.eval()
    # Generalized zero-shot learning
    if opt.gzsl:
        syn_feature, syn_label = generate_syn_feature(netG, data.unseenclasses, data.attribute, opt.syn_num)
        syn_seen_feature,syn_seen_label = generate_syn_feature(netG, data.seenclasses, data.attribute, opt.syn_num)
        train_X = torch.cat((data.train_feature, syn_feature), 0)
        train_Y = torch.cat((data.train_label, syn_label), 0)
        nclass = opt.nclass_all
        if opt.final_classifier == 'softmax':
            # cls0 = classifier_latent.CLASSIFIER(train_X, train_Y, data, nclass, opt.cuda, opt.classifier_lr, 0.5, 25, opt.syn_num, True)
            # print('unseen=%.4f, seen=%.4f, h=%.4f' % (cls0.acc_unseen, cls0.acc_seen, cls0.H))
            # if cls0.H > best_H_0:
            #     best_H_0 = cls0.H
            #     sio.savemat('syn_unseen.mat',{'unseen_feats':syn_feature.numpy(),'unseen_labels':syn_label.numpy()})
            #     torch.save(netG.state_dict(),
            #                './generator1/seen{0}_unseen{1}_H{2}.pkl'.format(cls0.acc_seen, cls0.acc_unseen, cls0.H))
            #     print('model saved!')

            cls = classifier2.CLASSIFIER(train_X, train_Y, data, nclass, opt.cuda, opt.classifier_lr, 0.5, 50, 2*opt.syn_num,True)
            print('unseen=%.4f, seen=%.4f, h=%.4f' % (cls.acc_unseen, cls.acc_seen, cls.H))
            if cls.H>best_H:
                best_H=cls.H
                torch.save(netG.state_dict(),'./generator/seen{0}_unseen{1}_H{2}.pkl'.format(cls.acc_seen,cls.acc_unseen,cls.H))
                print('model saved!!!!')
        elif opt.final_classifier == 'knn':
            clf = KNeighborsClassifier(n_neighbors=1)
            clf.fit(X=train_X.cuda().cpu(), y=train_Y)
            pred_Y_s = torch.from_numpy(clf.predict(data.test_seen_feature.cuda().cpu()))
            pred_Y_u = torch.from_numpy(clf.predict(data.test_unseen_feature.cuda().cpu()))
            acc_seen = compute_per_class_acc_gzsl(pred_Y_s, data.test_seen_label, data.seenclasses)
            acc_unseen = compute_per_class_acc_gzsl(pred_Y_u, data.test_unseen_label, data.unseenclasses)
            H = 2 * acc_seen * acc_unseen / (acc_seen + acc_unseen)
            print('unseen=%.4f, seen=%.4f, h=%.4f' % (acc_unseen, acc_seen, H))
            if H>=best_H:
                best_H=H
                print('model saved!!!')
    # Zero-shot learning
    else:
        syn_feature, syn_label = generate_syn_feature(netG, data.unseenclasses, data.attribute, opt.syn_num)
        cls = classifier2.CLASSIFIER(syn_feature, util.map_label(syn_label, data.unseenclasses), data,
                                     data.unseenclasses.size(0), opt.cuda, opt.classifier_lr, 0.5, 50, 2*opt.syn_num,
                                     False, epoch)
        if cls.acc>best_unseen:
            best_unseen=cls.acc
            print('best unseen acc is:',cls.acc)
    # del(cls)
    cls = None
    netG.train()
sio.savemat('orig_unseen.mat',{'unseen_feats':data.test_unseen_feature.numpy(),'unseen_labels':data.test_unseen_label.numpy()})