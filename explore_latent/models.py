import torch
import torch.nn as nn

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
        hidden = torch.cat((noise,att),dim=1)
        hidden = self.lrelu(self.fc1(hidden))
        latent_out = self.relu(self.fc2(hidden))
        return latent_out

class MLP_D(nn.Module):
    def __init__(self,opt):
        super(MLP_D, self).__init__()
        self.fc1 = nn.Linear(opt.latent_dim,opt.latent_dim*2)
        self.discriminator = nn.Linear(opt.latent_dim*2,1)
        self.classifier = nn.Linear(opt.latent_dim*2,opt.nclass)
        self.lrelu = nn.LeakyReLU(0.2,True)
        self.logic = nn.LogSoftmax(dim=1)
        self.apply(weights_init)

    def forward(self,x):
        hidden = self.lrelu(self.fc1(x))
        dis = self.discriminator(hidden)
        pred = self.logic(self.classifier(hidden))
        return dis,pred

class Mapping(nn.Module):
    def __init__(self,opt):
        super(Mapping, self).__init__()
        self.fc1 = nn.Linear(opt.visual_dim,opt.hidden_dim)
        self.fc2 = nn.Linear(opt.hidden_dim,opt.latent_dim)
        self.lrelu = nn.LeakyReLU(0.2,True)
        self.relu = nn.ReLU(True)
        self.apply(weights_init)

    def forward(self,x):
        hidden = self.lrelu(self.fc1(x))
        latent_out = self.relu(self.fc2(hidden))
        return latent_out