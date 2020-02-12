import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from generator import Generator
from discriminator import Discriminator
import numpy as np
import pickle as pkl

def getOneHot(arr, scale=10):
    dim = arr.size()
    val = torch.zeros(dim[0], scale)
    val.scatter_(1, arr.view(dim[0], 1), 1)
    return val


device = 'cuda' if torch.cuda.is_available() else 'cpu'

BATCH_SIZE = 128
EPOCH = 100
lr = 0.000001
momentum = 0.7

# Input normalization
transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
])

data = datasets.MNIST('data', train=True, download=True, transform=transform)
train = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)

D = Discriminator()
G = Generator()

D.to(device)
G.to(device)

if device == 'cuda':
    D= torch.nn.DataParallel(D)
    G = torch.nn.DataParallel(G)

d_optimizer = torch.optim.Adam(D.parameters(), lr, betas=(0.5, 0.99))
g_optimizer = torch.optim.Adam(G.parameters(), lr, betas=(0.5, 0.99))
loss = nn.BCELoss()

for epoch in range(EPOCH):
    for x, y in train:

        batch = x.size()[0]

        y_real = torch.ones(batch).to(device)
        y_fake = torch.zeros(batch).to(device)
        y_ = getOneHot(y).to(device)

        x = x.view(-1, 28*28).to(device)


        # Training Discriminator
        d_optimizer.zero_grad()
        D_result_real_data = D(x, y_).squeeze()

        d_real_loss = loss(D_result_real_data, y_real)

        z_ = torch.rand((batch, 100)).to(device)
        y_rand = (torch.rand(batch, 1) * 10).type(torch.LongTensor).to(device)
        y_rand_ = getOneHot(y_rand).to(device)

        G_result = G(z_, y_rand_)
        D_result_fake_data = D(G_result, y_rand_)

        d_fake_loss = loss(D_result_fake_data, y_fake)

        d_train_loss = d_real_loss + d_fake_loss
        d_train_loss.backward()
        d_optimizer.step()

        # Train Generator
        g_optimizer.zero_grad()
        z_ = torch.rand((batch, 100)).to(device)
        y_rand = (torch.rand(batch, 1) * 10).type(torch.LongTensor).to(device)
        y_rand_ = getOneHot(y_rand).to(device)

        G_result = G(z_, y_rand_)

        D_result_fake_data = D(G_result, y_rand_)
        g_loss = loss(D_result_fake_data, y_fake)

        g_loss.backward()
        g_optimizer.step()


    print('Epoch [{:5d}/{:5d}] | d_total_loss: {:6.4f} | g_loss: {:6.4f}'.format(epoch+1, EPOCH, d_train_loss.item(), g_loss.item()))

torch.save(G.state_dict(), './generator_state.mdl')
print('Model state saved....')
torch.save(G, './generator_model.mdl')
print('Model saved....')
