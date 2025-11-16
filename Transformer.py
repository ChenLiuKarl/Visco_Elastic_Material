import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
import math
import time
import datetime

import numpy as np
import scipy.io
import h5py

import matplotlib.pyplot as plt

from torch.optim.lr_scheduler import LambdaLR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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
        self._load_file()

    def _load_file(self):
        try:
            self.data = scipy.io.loadmat(self.file_path)
            self.old_mat = True
        except:
            self.data = h5py.File(self.file_path)
            self.old_mat = False

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
                x = x.to(device)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, scale=1.0):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # (L, D)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (L, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) * scale  # (1, L, D)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (B, L, D)
        return x + self.pe[:, :x.size(1), :]

def generate_causal_mask(sz: int, device):
    mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
    mask = mask.masked_fill(mask == 1, float("-inf"))
    return mask  # (L, L)


class Transformer(nn.Module):
    
    def __init__(
        self,
        input_dim=1,
        output_dim=1,
        seq_len=251,
        d_model=128,
        nhead=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=512,
        dropout=0.2,
        pe_scale=1.0,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model

        self.src_projection = nn.Linear(input_dim, d_model)
        self.tgt_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len + 1, scale=pe_scale)
        self.pos_decoder = PositionalEncoding(d_model, max_len=seq_len + 1, scale=pe_scale)
        self.dropout = nn.Dropout(dropout)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # (B, L, D)
        )

        self.output_projection = nn.Sequential(
            nn.Linear(d_model, output_dim),
            nn.Sigmoid()
        )

    def forward(self, src, tgt):
        # Embeddings + PE
        src_emb = self.src_projection(src) * math.sqrt(self.d_model)
        src_emb = self.pos_encoder(src_emb)
        src_emb = self.dropout(src_emb)

        tgt_emb = self.tgt_projection(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_decoder(tgt_emb)
        tgt_emb = self.dropout(tgt_emb)

        # Causal mask for decoder
        tgt_len = tgt_emb.size(1)
        tgt_mask = generate_causal_mask(tgt_len, tgt_emb.device)

        out = self.transformer(src_emb, tgt_emb, tgt_mask=tgt_mask)
        out = self.output_projection(out)
        return out
    

def lr_lambda(epoch):
    if epoch < WARMUP_EPOCHS:
        # Linear warmup
        return float(epoch + 1) / float(WARMUP_EPOCHS)
    else:
        # Cosine decay after warmup
        progress = float(epoch - WARMUP_EPOCHS) / float(epochs - WARMUP_EPOCHS)
        return 0.5 * (1.0 + math.cos(math.pi * progress))


def predict_sequence(model, src, out_len=251, start_token=None, device='cpu'):

    model.eval()
    src = src.to(device)
    B, _, in_dim = src.shape

    # Initialize decoder input
    if start_token is None:
        tgt = torch.zeros((B, 1, in_dim), device=device)
    else:
        if isinstance(start_token, (int, float)):
            tgt = torch.full((B, 1, in_dim), float(start_token), device=device)
        elif isinstance(start_token, torch.Tensor):
            if start_token.dim() == 2:  # (B, in_dim)
                start_token = start_token.unsqueeze(1)  # (B,1,in_dim)
            tgt = start_token.to(device)
        else:
            raise TypeError("start_token must be None, number, or torch.Tensor")

    outputs = []

    with torch.no_grad():
        for _ in range(out_len):
            # Forward pass
            out = model(src, tgt)
            
            # Get the last predicted value
            next_val = out[:, -1:, :] 
            outputs.append(next_val)

            # Append prediction to the decoder input
            tgt = torch.cat([tgt, next_val], dim=1)

    # Concatenate predictions into full sequence
    predicted = torch.cat(outputs, dim=1)
    return predicted


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

data_loader = MatReader(TRAIN_PATH, to_cuda=False)
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

net = Transformer(input_dim=1, output_dim=1).to(device)

n_params = sum(p.numel() for p in net.parameters() if p.requires_grad) #Calculate the number of training parameters
print('Number of parameters: %d' % n_params)

loss_func = LpLoss()

# Epoch number
epochs = 100
WARMUP_EPOCHS = 10

# Optimizer and learning drate scheduler
optimizer = torch.optim.AdamW(
    net.parameters(),
    lr=1e-4,
    betas=(0.9, 0.98),
    eps=1e-9
)
scheduler = LambdaLR(optimizer, lr_lambda)

# Batch size
b_size = 32

# Wrap training data in loader
x_train_enc = x_train_enc.unsqueeze(-1).to(device)
y_train_enc = y_train_enc.unsqueeze(-1).to(device)
x_test_enc = x_test_enc.unsqueeze(-1).to(device)
y_test_enc = y_test_enc.unsqueeze(-1).to(device)

train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(x_train_enc, y_train_enc),
    batch_size=b_size,
    shuffle=True
)

train_err = np.zeros((epochs))
test_err = np.zeros((epochs))
y_test_approx = torch.zeros(testsize, inputsize)

print("Start training for {} epochs...".format(epochs))
t_start = time.time()

for ep in range(epochs):

    net.train()
    
    train_loss = 0.0
    for x, y in train_loader:
        
        x, y = x.to(device), y.to(device)

        sos_token = torch.zeros((y.size(0), 1, y.size(2)), device=device)
        tgt_input = torch.cat([sos_token, y[:, :-1, :]], dim=1)
        tgt_output = y

        optimizer.zero_grad()
        
        out = net(x, tgt_input)
        loss = loss_func(out, tgt_output) 
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)

        train_loss += loss.item()
        optimizer.step()
    
    scheduler.step()

    net.eval()
    
    pred = predict_sequence(net, x_test_enc, out_len=251, device=device)
    t_loss = loss_func(pred, y_test_enc)
    test_loss = t_loss.item()

    train_err[ep] = train_loss/len(train_loader)
    test_err[ep]  = test_loss
    if ep % 10 == 0:
        print("epoch:{}, train loss:{}, test loss:{}".format(ep, train_loss/len(train_loader), test_loss))

print('Training Time: {}'.format(datetime.timedelta(seconds=int(time.time()-t_start))))
print('Train Loss: {}'.format(train_err[-1]))
print('Test Loss : {}'.format(test_err[-1]))

y_test_pred = b_normalizer.decode(pred)
y_test_approx = y_test_pred.cpu().detach().numpy()
y_test_np = y_test.cpu().detach().numpy()

plt.figure(1)
plt.plot(y_test_approx[0, :], 'r-', label='Predicted')
plt.plot(y_test_np[0, :], 'b-', label='True')
plt.title('Stress (True vs. Predicted) at Test Sample 0')
plt.legend()
plt.grid()
plt.savefig('stress0.png')

plt.figure(2)
plt.plot(y_test_approx[79, :], 'r-', label='Predicted')
plt.plot(y_test_np[79, :], 'b-', label='True')
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
