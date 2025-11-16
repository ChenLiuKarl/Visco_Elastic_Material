import torch
import torch.utils.data
import torch.nn as nn
import time
import datetime

import numpy as np
import scipy.io
import h5py

import matplotlib.pyplot as plt

class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        # Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        # Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h ** (self.d / self.p)) * torch.norm(x.view(num_examples, -1) - y.view(num_examples, -1), self.p,
                                                          1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms)
            else:
                return torch.sum(diff_norms / y_norms)

        return diff_norms / y_norms

    def forward(self, x, y):
        return self.rel(x, y)

    def __call__(self, x, y):
        return self.forward(x, y)
  

class MinMaxNormalizer(object):
    def __init__(self, x, eps=1e-5):
        super(MinMaxNormalizer, self).__init__()

        self.max = torch.max(x)
        self.min = torch.min(x)

    def encode(self, x):
        # Scale to [0, 1]
        x = (x - self.min) / (self.max - self.min)
        return x

    def decode(self, x):
        # Recover original scale
        x = x * (self.max - self.min) + self.min
        return x


class MatReader(object):
    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):
        super(MatReader, self).__init__()

        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float

        self.file_path = file_path

        self.data = None
        self.old_mat = None
        self._load_file()

    def _load_file(self):
        try:
            self.data = scipy.io.loadmat(self.file_path)
            self.old_mat = True
        except:
            self.data = h5py.File(self.file_path)
            self.old_mat = False

    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        x = self.data[field]

        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))

        if self.to_float:
            x = x.astype(np.float32)

        if self.to_torch:
            x = torch.from_numpy(x)

            if self.to_cuda:
                x = x.cuda()

        return x

    def set_cuda(self, to_cuda):
        self.to_cuda = to_cuda

    def set_torch(self, to_torch):
        self.to_torch = to_torch

    def set_float(self, to_float):
        self.to_float = to_float


class RNO(nn.Module):
    def __init__(self, hidden_size, layer_input, layer_hidden):
        super(RNO, self).__init__()

        self.layers = nn.ModuleList()
        for j in range(len(layer_input) - 1):
            self.layers.append(nn.Linear(layer_input[j], layer_input[j + 1]))
            if j != len(layer_input) - 1:
                self.layers.append(nn.SELU())

        self.hidden_layers = nn.ModuleList()
        self.hidden_size   = hidden_size

        for j in range(len(layer_hidden) - 1):
            self.hidden_layers.append(nn.Linear(layer_hidden[j], layer_hidden[j + 1]))
            if j != len(layer_hidden) - 1:
                self.hidden_layers.append(nn.SELU())

    def forward(self, input, output, hidden,dt):
        h0 = hidden
        h = torch.cat((output, hidden), 1)
        for _, m in enumerate(self.hidden_layers):
            h = m(h)

        h = h*dt + h0
        combined = torch.cat((output, (output-input)/dt, hidden), 1)
        x = combined
        for _, l in enumerate(self.layers):
            x = l(x)

        output = x.squeeze(1)
        hidden = h
        return output, hidden

    def initHidden(self,b_size):

        return torch.zeros(b_size, self.hidden_size)

TRAIN_PATH = 'viscodata_3mat.mat'

Ntotal     = 400
train_size = 320
test_start = 320

N_test = Ntotal-test_start

######### Preprocessing data ####################
temp = torch.zeros(Ntotal,1)

# Read data from the .mat file
F_FIELD = 'epsi_tol'

SIG_FIELD = 'sigma_tol'

data_loader = MatReader(TRAIN_PATH)
data_input  = data_loader.read_field(F_FIELD).contiguous().view(Ntotal, -1)
data_output  = data_loader.read_field(SIG_FIELD).contiguous().view(Ntotal, -1)

# Down sample the data to a coarser grid in time
s = 4

data_input  = data_input[:,0::s]
data_output = data_output[:,0::s]

inputsize   = data_input.size()[1]

print('Data loaded with input size {}, output size {}'.format(data_input.size(), data_output.size()))

# define train and test data
x_train = data_input[0:train_size,:]
y_train = data_output[0:train_size,:]

x_test = data_input[test_start:Ntotal,:]
y_test  = data_output[test_start:Ntotal,:]
testsize = x_test.shape[0]

a_normalizer = MinMaxNormalizer(x_train)
x_train_enc = a_normalizer.encode(x_train)
x_test_enc  = a_normalizer.encode(x_test)

b_normalizer = MinMaxNormalizer(y_train)
y_train_enc = b_normalizer.encode(y_train)
y_test_enc  = b_normalizer.encode(y_test)

# define the time increment dt in the RNO
dt = 1.0/(y_train.shape[1]-1)

n_hidden = 1

input_dim     = 1
output_dim    = 1

# Define RNO
layer_input = [input_dim+output_dim+n_hidden, 100, 100, output_dim]
layer_hidden = [input_dim+n_hidden, 100, 100, output_dim]
net = RNO(n_hidden, layer_input, layer_hidden)

n_params = sum(p.numel() for p in net.parameters() if p.requires_grad) #Calculate the number of training parameters
print('Number of parameters: %d' % n_params)

loss_func = LpLoss()

# Epoch number
epochs = 100

# Optimizer and learning drate scheduler
optimizer = torch.optim.Adam(net.parameters(), lr=0.003)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=120, gamma=0.85)

# Batch size
b_size = 40

# Wrap training data in loader
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train_enc, y_train_enc), batch_size=b_size,
                                           shuffle=True)
# Train neural net
T = inputsize
train_err = np.zeros((epochs))
test_err = np.zeros((epochs))
y_test_approx = torch.zeros(testsize, inputsize)

print("Start training for {} epochs...".format(epochs))
t_start = time.time()

for ep in range(epochs):
    
    train_loss = 0.0
    test_loss  = 0.0
    for x, y in train_loader:
        hidden = net.initHidden(b_size)
        optimizer.zero_grad()
        y_approx = torch.zeros(b_size,T)
        y_true  = y
        y_approx[:,0] = y_true[:,0]
        for i in range(1,T):
            y_approx[:,i], hidden = net(x[:,i].unsqueeze(1), x[:,i-1].unsqueeze(1), hidden,dt)

        loss = loss_func(y_approx,y_true)
        loss.backward()
        train_loss = train_loss + loss.item()

        optimizer.step()
        scheduler.step()

    with torch.no_grad():
        hidden_test = net.initHidden(testsize)
        y_test_approx[:,0] = y_test_enc[:,0]

        for j in range(1,T):
           y_test_approx[:, j], hidden_test = net(x_test_enc[:, j].unsqueeze(1), x_test_enc[:, j-1].unsqueeze(1), hidden_test,dt)
           
        t_loss = loss_func(y_test_approx,y_test_enc)
        test_loss = t_loss.item()

    train_err[ep] = train_loss/len(train_loader)
    test_err[ep]  = test_loss
    if ep % 10 == 0:
        print("epoch:{}, train loss:{}, test loss:{}".format(ep, train_loss/len(train_loader), test_loss))

print('Training Time: {}'.format(datetime.timedelta(seconds=int(time.time()-t_start))))
print('Train Loss: {}'.format(train_err[-1]))
print('Test Loss : {}'.format(test_err[-1]))

y_test_pred = b_normalizer.decode(y_test_approx)
y_test_approx = y_test_pred.cpu().detach().numpy()

plt.figure(1)
plt.plot(y_test_approx[0, :], 'r-', label='Predicted')
plt.plot(y_test[0, :], 'b-', label='True')
plt.title('Stress (True vs. Predicted) at Test Sample 0')
plt.legend()
plt.grid()
plt.savefig('stress0.png')

plt.figure(2)
plt.plot(y_test_approx[79, :], 'r-', label='Predicted')
plt.plot(y_test[79, :], 'b-', label='True')
plt.title('Stress (True vs. Predicted) at Test Sample 79')
plt.legend()
plt.grid()
plt.savefig('stress79.png')

plt.figure(3)
plt.plot(range(epochs), train_err, 'r-', label='Train Error')
plt.plot(range(epochs), test_err, 'b-', label='Test Error')
plt.legend()
plt.grid()
plt.savefig('error.png')

plt.show()
