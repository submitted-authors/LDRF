import torch.nn as nn
import torch

def weights_init(m):
    # if isinstance(m, nn.Linear):
    #     nn.init.xavier_normal_(m.weight)
    #     nn.init.constant_(m.bias, 0)
    # # 也可以判断是否为conv2d，使用相应的初始化方式
    # elif isinstance(m, nn.Conv2d):
    #     nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    #  # 是否为批归一化层
    # elif isinstance(m, nn.BatchNorm2d):
    #     nn.init.constant_(m.weight, 1)
    #     nn.init.constant_(m.bias, 0)
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class MLP_D(nn.Module):
    def __init__(self,opt):
        super(MLP_D, self).__init__()
        self.fc1 = nn.Linear(opt.resSize + opt.attSize, opt.ndh)
        self.fc2 = nn.Linear(opt.ndh, 1)
        self.fc3=nn.Linear(opt.ndh,opt.nclass_all)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.logic = nn.LogSoftmax(dim=1)
        self.apply(weights_init)

    def forward(self,x,att):
        #pred1 is discriminator.pred2 is classifier pred
        hidden=torch.cat((x,att),1)
        hidden = self.lrelu(self.fc1(hidden))

        pred1=self.fc2(hidden)
        pred2=self.logic(self.fc3(hidden))
        return pred1,pred2


# class MLP_CRITIC(nn.Module):
#     def __init__(self, opt):
#         super(MLP_CRITIC, self).__init__()
#         self.fc1 = nn.Linear(opt.resSize + opt.attSize, opt.ndh)
#         #self.fc2 = nn.Linear(opt.ndh, opt.ndh)
#         self.fc2 = nn.Linear(opt.ndh, 1)
#         self.lrelu = nn.LeakyReLU(0.2, True)
#
#         self.apply(weights_init)
#
#     def forward(self, x, att):
#         h = torch.cat((x, att), 1)
#         h = self.lrelu(self.fc1(h))
#         h = self.fc2(h)
#         return h


class MLP_G(nn.Module):
    def __init__(self, opt):
        super(MLP_G, self).__init__()
        self.fc1 = nn.Linear(opt.attSize + opt.nz + opt.latent_dim, opt.ngh)
        self.fc2 = nn.Linear(opt.ngh, opt.resSize)
        self.fc3 = nn.Linear(opt.resSize, opt.resSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        #self.prelu = nn.PReLU()
        self.relu = nn.ReLU(True).cuda()
        self.apply(weights_init)

    def forward(self, latent, noise, att, mean, var):
        h = torch.cat((att, noise, latent), 1)
        h=h.cuda()
        h = self.lrelu(self.fc1(h))
        h = self.relu(self.fc2(h))

        # gen_batch_mean = h.mean(dim=0)
        # gen_batch_var = h.var(dim=0)
        # h = (h-gen_batch_mean.unsqueeze(0))/(gen_batch_var.unsqueeze(0)+1e-5)
        # h = (h - mean.unsqueeze(0)) / (var.unsqueeze(0) + 1e-5)
        # h = self.relu(self.fc3(h))
        # h = h*var+mean
        return h


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
        return batch_mean,batch_var

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

