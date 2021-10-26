# import h5py
import numpy as np
import scipy.io as sio
import torch
from sklearn import preprocessing
import h5py


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


class Logger(object):
    def __init__(self, filename):
        self.filename = filename
        f = open(self.filename + '.log', "a")
        f.close()

    def write(self, message):
        f = open(self.filename + '.log', "a")
        f.write(message)
        f.close()


class DATA_LOADER(object):
    def __init__(self, opt):
        self.read_matdataset(opt)
        self.index_in_epoch = 0
        self.epochs_completed = 0

    def read_matdataset(self, opt):
        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".mat")
        feature = matcontent['features'].T
        label = matcontent['labels'].astype(int).squeeze() - 1
        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.class_embedding + "_splits.mat")
        # numpy array index starts from 0, matlab starts from 1
        trainval_loc = matcontent['trainval_loc'].squeeze() - 1
        train_loc = matcontent['train_loc'].squeeze() - 1
        val_unseen_loc = matcontent['val_loc'].squeeze() - 1
        test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
        test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1

        self.attribute = torch.from_numpy(matcontent['att'].T).float()

        scaler = preprocessing.MinMaxScaler()
        _train_feature = scaler.fit_transform(feature[trainval_loc])
        _test_seen_feature = scaler.transform(feature[test_seen_loc])
        _test_unseen_feature = scaler.transform(feature[test_unseen_loc])
        self.train_feature = torch.from_numpy(_train_feature).float()
        mx = self.train_feature.max()
        self.train_feature.mul_(1 / mx)
        self.train_label = torch.from_numpy(label[trainval_loc]).long()
        self.test_unseen_feature = torch.from_numpy(_test_unseen_feature).float()
        self.test_unseen_feature.mul_(1 / mx)
        self.test_unseen_label = torch.from_numpy(label[test_unseen_loc]).long()
        self.test_seen_feature = torch.from_numpy(_test_seen_feature).float()
        self.test_seen_feature.mul_(1 / mx)
        self.test_seen_label = torch.from_numpy(label[test_seen_loc]).long()

        self.seenclasses = torch.from_numpy(np.unique(self.train_label.numpy()))
        self.unseenclasses = torch.from_numpy(np.unique(self.test_unseen_label.numpy()))
        self.attribute_unseen = self.attribute[self.unseenclasses]
        self.attribute_seen = self.attribute[self.seenclasses]
        self.ntrain = self.train_feature.size()[0]
        self.ntrain_class = self.seenclasses.size(0)
        self.ntest_class = self.unseenclasses.size(0)
        self.train_class = self.seenclasses.clone()
        self.allclasses = torch.arange(0, self.ntrain_class + self.ntest_class).long()

        self.train_mapped_label = map_label(self.train_label, self.seenclasses)

        self.all_feature = torch.cat((self.train_feature, self.test_unseen_feature, self.test_seen_feature))
        self.both_feature = torch.cat((self.train_feature, self.test_unseen_feature))

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

    def next_batch(self, batch_size):
        idx = torch.randperm(self.ntrain)[0:batch_size]
        batch_feature = self.train_feature[idx]
        batch_label = self.train_label[idx]
        batch_att = self.attribute[batch_label]
        return batch_feature, batch_label, batch_att

    def next_batch_transductive(self, batch_size):
        idx = torch.randperm(self.ntrain)[0:batch_size]
        batch_seen_feature = self.train_feature[idx]
        batch_seen_label = self.train_label[idx]
        batch_seen_att = self.attribute[batch_seen_label]

        idx = torch.randperm(self.all_feature.shape[0])[0:batch_size]
        batch_both_feature = self.all_feature[idx]
        idx_both_att = torch.randint(0, self.attribute.shape[0], (batch_size,))
        batch_both_att = self.attribute[idx_both_att]

        return batch_seen_feature, batch_seen_label, batch_seen_att, batch_both_feature, batch_both_att

    def next_batch_transductive_both(self, batch_size):
        idx = torch.randperm(self.ntrain)[0:batch_size]
        batch_seen_feature = self.train_feature[idx]
        batch_seen_label = self.train_label[idx]
        batch_seen_att = self.attribute[batch_seen_label]

        idx = torch.randperm(self.both_feature.shape[0])[0:batch_size]
        batch_both_feature = self.both_feature[idx]
        idx_both_att = torch.randint(0, self.attribute.shape[0], (batch_size,))
        batch_both_att = self.attribute[idx_both_att]

        return batch_seen_feature, batch_seen_label, batch_seen_att, batch_both_feature, batch_both_att

    def next_batch_MMD(self, batch_size):
        # idx = torch.randperm(self.ntrain)[0:batch_size]
        index = torch.randint(self.seenclasses.shape[0], (2,))
        while index[0] == index[1]:
            index = torch.randint(self.seenclasses.shape[0], (2,))
        select_labels = self.seenclasses[index]
        X_features = self.train_feature[self.train_label == select_labels[0]]
        Y_features = self.train_feature[self.train_label == select_labels[1]]

        idx_X = torch.randperm(X_features.shape[0])[0:batch_size]
        X_features = X_features[idx_X]

        idx_Y = torch.randperm(Y_features.shape[0])[0:batch_size]
        Y_features = Y_features[idx_Y]

        return X_features, Y_features

    def next_batch_MMD_all(self):
        # idx = torch.randperm(self.ntrain)[0:batch_size]
        index = torch.randint(self.seenclasses.shape[0], (2,))
        while index[0] == index[1]:
            index = torch.randint(self.seenclasses.shape[0], (2,))
        select_labels = self.seenclasses[index]
        X_features = self.train_feature[self.train_label == select_labels[0]]
        Y_features = self.train_feature[self.train_label == select_labels[1]]

        return X_features, Y_features

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
