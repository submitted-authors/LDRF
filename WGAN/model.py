import torch.nn as nn
import torch
from torch.autograd import Variable

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
        self.fc2 = nn.Linear(opt.ngh, opt.resSize)
        self.fc3 = nn.Linear(opt.resSize, opt.resSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)
        self.apply(weights_init)

    def forward(self, noise, att, latent_x, mean, var):
        h = torch.cat((att, noise, latent_x), 1)
        h = self.lrelu(self.fc1(h))
        h = self.relu(self.fc2(h))

        # gen_batch_mean = h.mean(dim=0)
        # gen_batch_var = h.var(dim=0)
        # h = (h-gen_batch_mean.unsqueeze(0))/(gen_batch_var.unsqueeze(0)+1e-5)
        # # h = (h - mean.unsqueeze(0)) / (var.unsqueeze(0) + 1e-5)
        # h = self.relu(self.fc3(h))
        # h = h*gen_batch_var+gen_batch_mean
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

class MLP_CRITIC(nn.Module):
    def __init__(self, opt):
        super(MLP_CRITIC, self).__init__()
        self.resSize = opt.resSize
        self.fc1 = nn.Linear(opt.resSize + opt.attSize, opt.ndh)
        self.fc2 = nn.Linear(opt.resSize, 1)
        self.classifier = nn.Linear(opt.resSize, opt.nclass_seen)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.sigmoid = nn.Sigmoid()
        self.logic = nn.LogSoftmax(dim=1)
        self.apply(weights_init)

    def forward(self, x, att, train_G=False):
        mid = torch.cat((x, att), 1)
        mid = self.lrelu(self.fc1(mid))

        mus, stds = mid[:, :self.resSize], mid[:, self.resSize:]
        stds = self.sigmoid(stds)
        encoder_out = reparameter(mus, stds)#2048
        if not train_G:
            dis_out = self.fc2(encoder_out)
        else:
            dis_out = self.fc2(mus)
        pred = self.logic(self.classifier(mus))
        return dis_out, pred

def reparameter(mu,sigma):
    return (torch.randn_like(mu) *sigma) + mu

class ClassStandardization(nn.Module):
    """
    Class Standardization procedure from the paper.
    Conceptually, it is equivalent to nn.BatchNorm1d with affine=False,
    but for some reason nn.BatchNorm1d performs slightly worse.
    """

    def __init__(self, feat_dim: int):
        super().__init__()
        self.running_mean = nn.Parameter(torch.zeros(feat_dim), requires_grad=False)
        self.running_var = nn.Parameter(torch.ones(feat_dim), requires_grad=False)

    def forward(self, class_feats):
        """
        Input: class_feats of shape [num_classes, feat_dim]
        Output: class_feats (standardized) of shape [num_classes, feat_dim]
        """
        if self.training:
            batch_mean = class_feats.mean(dim=0)
            batch_var = class_feats.var(dim=0)

            # Normalizing the batch
            result = (class_feats - batch_mean.unsqueeze(0)) / (batch_var.unsqueeze(0) + 1e-5)

            # Updating the running mean/std
            self.running_mean.data = 0.9 * self.running_mean.data + 0.1 * batch_mean
            self.running_var.data = 0.9 * self.running_var.data + 0.1 * batch_var
        else:
            # Using accumulated statistics
            # Attention! For the test inference, we cant use batch-wise statistics,
            # only the accumulated ones. Otherwise, it will be quite transductive
            result = (class_feats - self.running_mean.unsqueeze(0)) / (self.running_var.unsqueeze(0) + 1e-5)
        batch_mean = Variable(batch_mean.cuda())
        batch_var = Variable(batch_var.cuda())
        return batch_mean, batch_var