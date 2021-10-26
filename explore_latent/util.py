import numpy as np
import scipy.io as sio
import torch
from sklearn import preprocessing

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def map_label(label, classes):
    mapped_label = torch.LongTensor(label.size())
    for i in range(classes.size(0)):
        mapped_label[label == classes[i]] = i
    return mapped_label

class DATA_LOADER(object):
    def __init__(self, opt):
        self.read_matdataset(opt)
        self.index_in_epoch = 0
        self.epochs_completed = 0

    def read_matdataset(self, opt):
        #load all features and labels
        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".mat")
        feature = matcontent['features'].T
        label = matcontent['labels'].astype(int).squeeze() - 1

        #load all trainval indexs
        matcontent_index = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.class_embedding + "_splits.mat")
        # numpy array index starts from 0, matlab starts from 1
        trainval_loc = matcontent_index['trainval_loc'].squeeze() - 1
        test_seen_loc = matcontent_index['test_seen_loc'].squeeze() - 1

        self.attribute = torch.from_numpy(matcontent_index['att'].T).float()#[50,85]

        scaler = preprocessing.MinMaxScaler()
        all_feature = scaler.fit_transform(feature[trainval_loc])
        all_test_feature = scaler.transform(feature[test_seen_loc])

        self.all_feature = torch.from_numpy(all_feature).float()#[19832,2048]
        mx = self.all_feature.max()
        self.all_feature.mul_(1 / mx)
        self.all_label = torch.from_numpy(label[trainval_loc]).long()#[19832],0-based from [ 0  1  2  3  4  5  7  9 10 11 12 13 14 15 16 17 18 19 20 21 24 25 26 27 28 31 32 34 35 36 37 38 39 41 42 43 44 45 47 48]

        self.all_test_feature = torch.from_numpy(all_test_feature).float()#[4958,2048]
        self.all_test_feature.mul_(1 / mx)
        self.all_test_label = torch.from_numpy(label[test_seen_loc]).long()#[4958],from [ 0  1  2  3  4  5  7  9 10 11 12 13 14 15 16 17 18 19 20 21 24 25 26 27 28 31 32 34 35 36 37 38 39 41 42 43 44 45 47 48]

        #we split train set in train set and test set again for determining parameters
        self.seenclasses = torch.from_numpy(np.unique(self.all_label.numpy()))[0:opt.reseen_classes]#[30]
        self.unseenclasses = torch.from_numpy(np.unique(self.all_label.numpy()))[opt.reseen_classes:opt.reseen_classes+opt.reunseen_classes]#[10]

        #resplit train set
        self.retrain_indexs = np.array([label in self.seenclasses for label in self.all_label])
        self.train_feature = self.all_feature[self.retrain_indexs]#[14194,2048]
        self.train_label = self.all_label[self.retrain_indexs]#[14194]

        #resplit test seen and unseen set
        self.retest_seen_indexs = np.array([label in self.seenclasses for label in self.all_test_label])
        self.test_seen_feature = self.all_test_feature[self.retest_seen_indexs]#[3513,2048]
        self.test_seen_label = self.all_test_label[self.retest_seen_indexs]#[3513]

        self.retest_unseen_indexs = np.array([label in self.unseenclasses for label in self.all_test_label])
        self.test_unseen_feature = self.all_test_feature[self.retest_unseen_indexs]#[1445,2048]
        self.test_unseen_label = self.all_test_label[self.retest_unseen_indexs]#[1445]

        self.attribute_unseen = self.attribute[self.unseenclasses]#[10,85]
        self.attribute_seen = self.attribute[self.seenclasses]#[30,85]

        self.ntrain = self.train_feature.size()[0]#14194
        self.ntrain_class = self.seenclasses.size(0)#30
        self.ntest_class = self.unseenclasses.size(0)#10

        self.train_class = self.seenclasses.clone()
        self.allclasses = torch.arange(0, self.ntrain_class + self.ntest_class).long()#from 0-39

        self.train_mapped_label = map_label(self.train_label, self.seenclasses)#convert labels to 0-29

        self.all_included_feature = torch.cat((self.train_feature, self.test_unseen_feature, self.test_seen_feature))#[30475,2048]
        self.both_feature = torch.cat((self.train_feature, self.test_unseen_feature))#[25517,2048]

    def next_batch(self, batch_size):
        idx = torch.randperm(self.ntrain)[0:batch_size]
        batch_feature = self.train_feature[idx]
        batch_label = self.train_label[idx]
        batch_att = self.attribute[batch_label]
        return batch_feature, batch_label, batch_att

    def next_batch_one_class(self, batch_size):
        if self.index_in_epoch == self.ntrain_class:
            self.index_in_epoch = 0
            perm = torch.randperm(self.ntrain_class)
            self.train_class[perm] = self.train_class[perm]

        iclass = self.train_class[self.index_in_epoch]
        idx = self.train_label.eq(iclass).nonzero().squeeze()
        perm = torch.randperm(idx.size(0))
        idx = idx[perm]
        iclass_feature = self.train_feature[idx]
        iclass_label = self.train_label[idx]
        self.index_in_epoch += 1
        return iclass_feature[0:batch_size], iclass_label[0:batch_size], self.attribute[iclass_label[0:batch_size]]

    def next_batch_unseenatt(self, batch_size, unseen_batch_size):
        idx = torch.randperm(self.ntrain)[0:batch_size]
        batch_feature = self.train_feature[idx]
        batch_label = self.train_label[idx]
        batch_att = self.attribute[batch_label]

        # idx = torch.randperm(data)[0:batch_size]
        idx_unseen = torch.randint(0, self.unseenclasses.shape[0], (unseen_batch_size,))
        unseen_label = self.unseenclasses[idx_unseen]
        unseen_att = self.attribute[unseen_label]
        return batch_feature, batch_label, batch_att, unseen_label, unseen_att

    # select batch samples by randomly drawing batch_size classes
    def next_batch_uniform_class(self, batch_size):
        batch_class = torch.LongTensor(batch_size)
        for i in range(batch_size):
            idx = torch.randperm(self.ntrain_class)[0]
            batch_class[i] = self.train_class[idx]

        batch_feature = torch.FloatTensor(batch_size, self.train_feature.size(1))
        batch_label = torch.LongTensor(batch_size)
        batch_att = torch.FloatTensor(batch_size, self.attribute.size(1))
        for i in range(batch_size):
            iclass = batch_class[i]
            idx_iclass = self.train_label.eq(iclass).nonzero().squeeze()
            idx_in_iclass = torch.randperm(idx_iclass.size(0))[0]
            idx_file = idx_iclass[idx_in_iclass]
            batch_feature[i] = self.train_feature[idx_file]
            batch_label[i] = self.train_label[idx_file]
            batch_att[i] = self.attribute[batch_label[i]]
        return batch_feature, batch_label, batch_att