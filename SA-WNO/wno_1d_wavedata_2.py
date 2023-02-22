"""
This code belongs to the paper:
-- Tripura, T., & Chakraborty, S. (2022). Wavelet neural operator: a neural 
   operator for parametric partial differential equations. arXiv preprint arXiv:2205.02191.
   
This code is for 2-D Allen-Cahn equation (time-independent problem).
"""

# from IPython import get_ipython
# get_ipython().magic('reset -sf')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt
from torch.autograd import Variable
import os

from timeit import default_timer
from utilities3 import *
from pytorch_wavelets import DWT1D, IDWT1D

torch.manual_seed(0)
np.random.seed(0)

# %%
""" Def: 1d Wavelet layer """
class WaveConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, level, dummy, wavelet):
        super(WaveConv1d, self).__init__()

        """
        1D Wavelet layer. It does Wavelet Transform, linear transform, and
        Inverse Wavelet Transform.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.level = level
        self.wavelet = wavelet  #'db20'
        self.dwt_ = DWT1D(wave=self.wavelet, J=self.level, mode='symmetric').to(dummy.device)
        self.mode_data, self.mode_coeff = self.dwt_(dummy)
        self.modes1 = self.mode_data.shape[-1]
        self.modes2 = self.mode_coeff[-2].shape[-1]
        self.modes3 = self.mode_coeff[-3].shape[-1]
        
        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes2))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes3))

    # Convolution
    def mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute single tree Discrete Wavelet coefficients using some wavelet     
        dwt = DWT1D(wave=self.wavelet, J=self.level, mode='symmetric').to(x.device)
        x_ft, x_coeff = dwt(x)
        
        # Multiply the final low pass and high pass coefficients
        out_ft = torch.zeros(batchsize, self.out_channels, x_ft.shape[-1],  device=x.device)
        out_ft = self.mul1d(x_ft, self.weights1)
        x_coeff[-1] = self.mul1d(x_coeff[-1].clone(), self.weights2)
        x_coeff[-2] = self.mul1d(x_coeff[-2].clone(), self.weights3)
        x_coeff[-3] = self.mul1d(x_coeff[-3].clone(), self.weights4)
        
        # Reconstruct the signal
        idwt = IDWT1D(wave=self.wavelet, mode='symmetric').to(x.device)
        x = idwt((out_ft, x_coeff))        
        return x
        
""" The forward operation """
class WNO1d(nn.Module):
    def __init__(self, width, level, dummy_data):
        super(WNO1d, self).__init__()

        """
        The WNO network. It contains 4 layers of the Wavelet integral layer.
        1. Lift the input using v(x) = self.fc0 .
        2. 4 layers of the integral operators v(+1) = g(K(.) + W)(v).
            W is defined by self.w_; K is defined by self.conv_.
        3. Project the output of last layer using self.fc1 and self.fc2.
        
        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.level = level
        self.width = width
        self.dummy_data = dummy_data
        self.padding = 2 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(2, self.width)

        self.conv0 = WaveConv1d(self.width, self.width, self.level, self.dummy_data, 'db24')
        self.conv1 = WaveConv1d(self.width, self.width, self.level, self.dummy_data, 'db24')
        self.conv2 = WaveConv1d(self.width, self.width, self.level, self.dummy_data, 'db24')
        self.conv3 = WaveConv1d(self.width, self.width, self.level, self.dummy_data, 'db24')
        self.conv4 = WaveConv1d(self.width, self.width, self.level, self.dummy_data, 'db24')
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)
        self.w4 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)
        # x = F.pad(x, [0,self.padding]) # do padding, if required

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)
        
        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2
        x = F.gelu(x)
        
        x1 = self.conv4(x)
        x2 = self.w4(x)
        x = x1 + x2

        # x = x[..., :-self.padding] # remove padding, when required
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        # The grid of the solution
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, 100, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)

class PARAM(nn.Module):
    def __init__(self, dim):
        super(PARAM, self).__init__()
        # self.lam = nn.Parameter(torch.randn(dim))
        self.lam = nn.Parameter(torch.ones(dim))
        
        def forward(self):
            pass
        
# %%
""" Model configurations """

PATH = 'Data/data.mat'
save_index = 1 
ntrain = 14400
nvali = 100
ntest = 100

batch_size_train = 50
batch_size_vali = 50
learning_rate = 0.001

epochs = 300
step_size = 50
gamma = 0.5

level = 6
width = 48

h = 1000
s = h

# %%
""" Read data """
reader = MatReader(PATH)

x_train = reader.read_field('f_train')[-ntrain:,:]
y_train = reader.read_field('u_train')[-ntrain:,:]

x_vali = reader.read_field('f_train')[-nvali:,:]
y_vali = reader.read_field('u_train')[-nvali:,:]

x_test = reader.read_field('f_test')[:ntest,:]
y_test = reader.read_field('u_test')[:ntest,:]

x_train = x_train.reshape(ntrain,s,1)
x_vali = x_vali.reshape(nvali,s,1)
x_test = x_test.reshape(ntest,s,1)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train),
                                           batch_size=batch_size_train, shuffle=True)
vali_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_vali, y_vali),
                                          batch_size=batch_size_vali, shuffle=False)

# %%
""" The model definition """
model = WNO1d(width, level, x_train[0:1].permute(0,2,1)).to(device)
model_param = PARAM(h).to(device)
print(count_params(model))

# lam = nn.Parameter(torch.ones(dim))

""" Training and testing """
sub_model_params = [{
        "params": model_param.parameters(), "weight_decay": 1e-6, "lr": -1e-2
    }]
    
optimizer1 = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6)
optimizer2 = torch.optim.Adam(sub_model_params)
scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=step_size, gamma=gamma)
# scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=step_size, gamma=gamma)

epoch_loss = torch.zeros(epochs)
lambda_epoch = torch.zeros(epochs,h)

myloss = LpLoss(size_average=False)
train_error = np.zeros((epochs, 1))
train_loss = np.zeros((epochs, 1))
vali_error = np.zeros((epochs, 1))
vali_loss = np.zeros((epochs, 1))
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_mse = 0
    train_l2 = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer1.zero_grad()
        optimizer2.zero_grad()
        
        out = model(x).view(batch_size_train, -1)
        
        # lamb = torch.sin(torch.pi*model_param.lam*torch.exp(model_param.lam))
        lamb = torch.sin(2*np.pi*model_param.lam)
        
        mse = torch.sum(torch.mean(torch.einsum('ij,j->ij',torch.square(out - y),lamb**2), dim=0))
        
        l2 = myloss(out, y)
        mse.backward() # l2 relative loss

        optimizer1.step()
        optimizer2.step()
        train_mse += mse.item()
        train_l2 += l2.item()

    scheduler1.step()
    # scheduler2.step()

    model.eval()
    test_mse = 0.0
    test_l2 = 0.0
    with torch.no_grad():
        for x, y in vali_loader:
            x, y = x.to(device), y.to(device)

            out = model(x).view(batch_size_vali, -1)
            mse=F.mse_loss(out.view(batch_size_vali, -1), y.view(batch_size_vali, -1), reduction='mean')
            test_l2 += myloss(out, y).item()
            test_mse += mse.item()

    train_mse /= len(train_loader)
    test_mse /= len(vali_loader)
    train_l2 /= len(train_loader)
    test_l2 /= len(vali_loader)
    train_error[ep,0] = train_l2
    vali_error[ep,0] = test_l2
    train_loss[ep,0] = train_mse
    vali_loss[ep,0] = test_mse
    epoch_loss[ep] = train_mse

    lambda_epoch[ep] = model_param.lam.detach().cpu()

    t2 = default_timer()
    print(ep, t2-t1, train_mse, train_l2, test_l2)

# %%
""" Prediction """
pred = torch.zeros(y_test.shape)
index = 0
test_e = torch.zeros(y_test.shape[0])
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test),
                                          batch_size=1, shuffle=False)
test_l2 = 0
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)

        out = model(x).view(1,-1)
        pred[index] = out
        # test_l2 += myloss(out.view(1, -1), y.view(1, -1)).item()
        test_l2 += (F.mse_loss(out, y)).item()
        test_e[index] = test_l2
        #print(index, test_l2)
        index = index + 1

print('Mean Error:', torch.mean(test_e))

# ====================================
# saving settings
# ====================================
current_directory = os.getcwd()
case = "Case_"
folder_index = str(save_index)

results_dir = "/" + case + folder_index +"/"
save_results_to = current_directory + results_dir
if not os.path.exists(save_results_to):
    os.makedirs(save_results_to)

save_models_to = save_results_to +"model/"

if not os.path.exists(save_models_to):
    os.makedirs(save_models_to)
    
x = np.linspace(0, epochs-1, epochs)
np.savetxt(save_results_to+'/epoch.txt', x)
np.savetxt(save_results_to+'/train_loss.txt', train_loss)
np.savetxt(save_results_to+'/vali_loss.txt', vali_loss)
np.savetxt(save_results_to+'/train_error.txt', train_error)
np.savetxt(save_results_to+'/vali_error.txt', vali_error)    

torch.save(model, save_models_to+'Wave_states')

scipy.io.savemat(save_results_to+'wave_states_test.mat', 
                     mdict={'y_test': y_test.numpy(), 
                            'y_pred': pred.cpu().numpy()})  

# %%
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 20

""" Plotting """ # for paper figures please see 'WNO_testing_(.).py' files
cases = [0,5,10,15]
index = 0
figure7 = plt.figure(figsize = (14, 8))
plt.subplots_adjust(hspace=0.3)
for i in range(len(cases)):
    plt.subplot(2,2,index+1)

    plt.plot(y_test[cases[index], :].numpy(), label='Actual')
    plt.plot(pred[cases[index],:].numpy(), 'k', label='Prediction')
    plt.title('Case-{}'.format(cases[i]), color='r')
    if i == len(cases)-1:
        plt.legend(ncol=2)
    plt.margins(0)
    index = index+1

# %%
figure8 = plt.figure(figsize = (10, 8))
plt.subplots_adjust(hspace=0.3)

plt.subplot(2,1,1)
plt.plot(epoch_loss)
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.yscale('log')
plt.margins(0)

plt.subplot(2,1,2)
plt.plot(lambda_epoch)
plt.xlabel('Epoch')
plt.ylabel('$\lambda$')
plt.margins(0)

# %%
"""
For saving the trained model and prediction data
"""

# figure7.savefig('Results_0269.png', format='png', dpi=300, bbox_inches='tight')
# figure8.savefig('Lambda_0269.png', format='png', dpi=300, bbox_inches='tight')

# torch.save(model, 'model/mse_0269')
# scipy.io.savemat('pred/epoch_loss_0269.mat', mdict={'epoch_loss': epoch_loss.cpu().numpy()})
# scipy.io.savemat('pred/lambda_epoch_0269.mat', mdict={'lambda_epoch': lambda_epoch.cpu().numpy()})