import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as tg
import torch_scatter

from ase import Atom
from ase.data import atomic_masses

from tqdm import tqdm
from utils.data import XANES, Process, bar_format
from utils.e3nn import Network

default_dtype = torch.float64
torch.set_default_dtype(default_dtype)
device = "cuda:0" if torch.cuda.is_available() else 'cpu'

if not os.path.exists('images/'):
    os.makedirs('images/')

if not os.path.exists('models/'):
    os.makedirs('models/')

#Stating the element of focus
element = 'Cu'

# Load data
data_file1 = 'data/' + element + '_XANES.json'
xanes = XANES()
xanes.load_data(data_file1)
xanes.data

#Removing XFAS data
outliers_den = []
for i in xanes.data.index:
  if (max(xanes.data['spectrum'][i]['x']) - min(xanes.data['spectrum'][i]['x']))/len(xanes.data['spectrum'][i]['x']) > 1:
    outliers_den += [xanes.data['formula_pretty'][i]]
    xanes.data = xanes.data.drop(index = i)
xanes.data.index = np.arange(0,len(xanes.data))
print(len(outliers_den))

#Removing y outliers
outliers_y = []
for i in xanes.data.index:
  if min(xanes.data['spectrum'][i]['y']) < 0 or max(xanes.data['spectrum'][i]['y']) > 100:
    outliers_y += [xanes.data['formula_pretty'][i]]
    xanes.data = xanes.data.drop(index = i)
xanes.data.index = np.arange(0,len(xanes.data))
print(len(outliers_y))

#Removing outlier elements from our dataset
#Only those with less than 20 examples each
#We run through the data multiple time though because sometimes removing data can
#move an element to the outlier range so we must go through it multiple times
#This does basically the same things as setting a species_min but this way we get a list of everything removed
#and every element not included
elements = ('H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','Si','P','S','Cl','Ar','K','Ca','Sc','Ti',
              'V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga','Ge','As','Se','Br','Kr','Rb','Sr','Y','Zr','Nb','Mo','Tc','Ru',
              'Rh','Pd','Ag','Cd','In','Sn','Sb','Te','I','Xe','Cs','Ba','La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb',
              'Dy','Ho','Er','Tm','Yb','Lu','Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg','Tl','Pb','Bi','Po','At','Rn',
              'Fr','Ra','Ac','Th','Pa','U','Np','Pu','Am','Cm','Bk','Cf','Es','Fm')

outliers_e = []
removed_elements = []
min_elements = 20
for t in range(5):
  for e in elements:
    elemental_counter = 0
    for i in xanes.data.index:
      if e in xanes.data['formula_pretty'][i]:
        elemental_counter += 1
    if elemental_counter < min_elements:
      if t == 4: 
        removed_elements += [e]
      for i in xanes.data.index:
        if e in xanes.data['formula_pretty'][i]:
          outliers_e += [xanes.data['formula_pretty'][i]]
          xanes.data = xanes.data.drop(index = i)
xanes.data.index = np.arange(0,len(xanes.data))
print("The number of elemental outliers is", len(outliers_e))
print("The list of non-included elements is", removed_elements)

#Analyze the max and the mins of the data as well as the density
minisy = np.zeros(len(xanes.data))
maxisy = np.zeros(len(xanes.data))
minisx = np.zeros(len(xanes.data))
maxisx = np.zeros(len(xanes.data))
densisx = np.zeros(len(xanes.data))
for i in xanes.data.index:
  minisy[i] = min(xanes.data['spectrum'][i]['y'])
  maxisy[i] = max(xanes.data['spectrum'][i]['y'])
  minisx[i] = min(xanes.data['spectrum'][i]['x'])
  maxisx[i] = max(xanes.data['spectrum'][i]['x'])
  densisx[i] = (max(xanes.data['spectrum'][i]['x']) - min(xanes.data['spectrum'][i]['x']))/len(xanes.data['spectrum'][i]['x'])

#Creating a range that removes only 1% of total data
xmin = np.percentile(minisx,99.5)
xmax = np.percentile(maxisx,.5)
min = np.min
print(xmin,xmax)

#Removing  x outliers
outliers_x = []
for i in xanes.data.index:
  if min(xanes.data['spectrum'][i]['x']) > xmin or max(xanes.data['spectrum'][i]['x']) < xmax:
    outliers_x += [xanes.data['formula_pretty'][i]]
    xanes.data = xanes.data.drop(index = i)
xanes.data.index = np.arange(0,len(xanes.data))
print(len(outliers_x))


# Interpolating the XANES data to uniform energy bins
x_new = np.linspace(xmin,xmax,len(xanes.data['spectrum'][0]['y']))
for i in xanes.data.index:
  x_old = xanes.data['spectrum'][i]['x']
  y_old = xanes.data['spectrum'][i]['y']
  y_new = np.interp(x_new,x_old,y_old)
  xanes.data['spectrum'][i]['x'] = x_new
  xanes.data['spectrum'][i]['y'] = y_new
xanes.data

# Enforce a minimum number of examples of each specie
species_min = 0
xanes.get_species_counts()

# Lattice parameter statistics
xanes.get_lattice_parameters()

# Get species
species = sorted(list(set(xanes.data['species'].sum())))
n_species = list(np.unique(xanes.data['species'].sum(), return_counts=True)[1])
Z_max = max([Atom(k).number for k in species])
print(Z_max)

# One-hot encoding atom type and mass
type_encoding = {}
mass_specie = []

for Z in tqdm(range(1, Z_max + 1), bar_format=bar_format):
    specie = Atom(Z)
    type_encoding[specie.symbol] = Z - 1
    mass_specie.append(atomic_masses[Z])

type_onehot = torch.eye(len(type_encoding))
mass_onehot = torch.diag(torch.tensor(mass_specie))

# Process data into input descriptors
process = Process(species, Z_max, type_encoding, type_onehot, mass_onehot, default_dtype)

r_max = 5.     # cutoff radius
tqdm.pandas(desc='Building data', bar_format=bar_format)
xanes.data['input'] = xanes.data.progress_apply(lambda x: process.build_data(x, r_max), axis=1)

#Creating a constant test set with random train and val sets
test_size = .2
backup_seed = np.random.randint(10000) + 50
np.random.seed(42)
num_list = np.linspace(0,len(xanes.data)-1,len(xanes.data))
np.random.shuffle(num_list)

train_val_index = []
test_index = []
val_index = []
train_index= []
for i in range(len(xanes.data)):
  if i/len(xanes.data) < test_size:
    test_index += [int(num_list[i])]
  else: 
    train_val_index += [int(num_list[i])]

np.random.seed(backup_seed)
np.random.shuffle(train_val_index)

for i in range(len(train_val_index)):
  if i/len(train_val_index) < test_size:
    val_index += [train_val_index[i]]
  else: 
    train_index += [train_val_index[i]]

print("The training size is", len(train_index))
print("The validation size is", len(val_index))
print("The test size is", len(test_index))

process.idx_train = train_index
process.idx_valid = val_index
process.idx_test = test_index

# Alternative Train/valid/test split that ensures proportional distribution of elements
#test_size = 0.2
#fig = process.train_valid_test_split(xanes.data, valid_size=test_size, test_size=test_size, plot=True)
#fig.savefig('images/train_valid_test_split.svg', bbox_inches='tight')

# Calculate average number of neighbors
process.get_neighbors(xanes.data)

# Format dataloaders
#You can replace this
batch_size = 64
dataloader_train = tg.loader.DataLoader(xanes.data.iloc[process.idx_train]['input'].tolist(), batch_size=batch_size,
                                        shuffle=True)
dataloader_valid = tg.loader.DataLoader(xanes.data.iloc[process.idx_valid]['input'].tolist(), batch_size=batch_size)
dataloader_test = tg.loader.DataLoader(xanes.data.iloc[process.idx_test]['input'].tolist(), batch_size=batch_size)

class E3NN(Network):
    def __init__(self, in_dim, out_dim, emb_dim, num_layers, mul, lmax, max_radius, num_basis, radial_layers,
                 radial_neurons, num_neighbors):
        kwargs = {'reduce_output': False,
                  'irreps_in': str(emb_dim)+"x0e",
                  'irreps_out': str(out_dim)+"x0e",
                  'irreps_node_attr': str(emb_dim)+"x0e",
                  'layers': num_layers,
                  'mul': mul,
                  'lmax': lmax,
                  'max_radius': max_radius,
                  'number_of_basis': num_basis,
                  'radial_layers': radial_layers,
                  'radial_neurons': radial_neurons,
                  'num_neighbors': num_neighbors
                 }
        super().__init__(**kwargs)

        # definitions
        self.cmap = plt.get_cmap('plasma')
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.emb_dim = emb_dim
        self.num_layers = num_layers
        self.mul = mul
        self.lmax = lmax
        self.max_radius = max_radius
        self.num_basis = num_basis
        self.radial_layers = radial_layers
        self.radial_neurons = radial_neurons
        self.num_neighbors = num_neighbors

        self.model_name = 'e3nn-xanes_' + '_'.join(i + str(int(j)) for (i,j) in zip(
            ['emb', 'layers', 'mul', 'lmax', 'rmax', 'nbasis', 'rlayers', 'rneurons'],
            [emb_dim, num_layers, mul, lmax, max_radius, num_basis, radial_layers, radial_neurons]))

        # embedding
        self.emb_x = nn.Sequential(
            nn.Linear(in_dim, emb_dim),
            nn.ReLU()
        )

        self.emb_z = nn.Sequential(
            nn.Linear(in_dim, emb_dim),
            nn.Tanh()
        )


    def forward(self, data):
        data['x'] = self.emb_x(data['x_in'])
        data['z'] = self.emb_z(data['z_in'])
        x = super().forward(data)[0]

        # aggregate
        if 'batch' in data:
            batch = data['batch']
        else:
            batch = data['pos'].new_zeros(data['pos'].shape[0], dtype=torch.long)

        y = torch_scatter.scatter_mean(x, batch, dim=0)
        return y


    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


    def loss(self, y_pred, y_true):
        return torch.mean((y_pred-y_true)**2)


    def checkpoint(self, dataloader, device):
        self.eval()

        loss_cum = 0.
        with torch.no_grad():
            for j, d in enumerate(dataloader):
                d.to(device)
                y_pred = self.forward(d)

                loss = self.loss(y_pred, d.y).cpu()
                loss_cum += loss.detach().item()

        return loss_cum/len(dataloader)


    def fit(self, opt, dataloader_train, dataloader_valid, history, s0, max_iter=10, device="cpu", scheduler=None):
        chkpt = 1

        for step in range(max_iter):
            self.train()

            loss_cum = 0.
            start_time = time.time()

            for j, d in enumerate(dataloader_train):
                d.to(device)
                y_pred = self.forward(d)

                loss = self.loss(y_pred, d.y).cpu()
                loss_cum += loss.detach().item()

                print(f"Iteration {step+1:5d}    batch {j+1:5d} / {len(dataloader_train):5d}   " +
                      f"batch loss = {loss.data:.4e}", end="\r", flush=True)

                opt.zero_grad()
                loss.backward()
                opt.step()

            if scheduler is not None:
                scheduler.step()

            end_time = time.time()
            wall = end_time - start_time

            if (step+1)%chkpt == 0:
                print(f"Iteration {step+1:5d}    batch {j+1:5d} / {len(dataloader_train):5d}   " +
                      f"epoch loss = {loss_cum/len(dataloader_train):.4e}")

                loss_valid = self.checkpoint(dataloader_valid, device)
                loss_train = self.checkpoint(dataloader_train, device)

                history.append({
                    'step': step + s0,
                    'wall': wall,
                    'batch': {
                        'loss': loss.item(),
                    },
                    'valid': {
                        'loss': loss_valid,
                    },
                     'train': {
                         'loss': loss_train,
                     },
                })

                yield {
                    'history': history,
                    'state': self.state_dict(),
                    'optimizer': opt.state_dict(),
                    'scheduler': scheduler.state_dict() if scheduler else None
                }

args_enn = {'in_dim': Z_max,
            'out_dim': xanes.data.iloc[0]['input'].y.shape[-1],
            'emb_dim': 64,
            'num_layers': 2,
            'mul': 32,
            'lmax': 2,
            'max_radius': r_max,
            'num_basis': 10,
            'radial_layers': 1,
            'radial_neurons': 100,
            'num_neighbors': process.n_train.mean(),
           }

enn = E3NN(**args_enn).to(device)
opt = torch.optim.Adam(enn.parameters(), lr=1e-3)
scheduler = None #torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.99)

model_num = 0
model_path = 'models/' + enn.model_name + '_' + str(model_num) + element + '.torch'

print(model_path)
#print(enn)
print('Number of parameters:', enn.count_parameters())

resume = False

if resume:
    saved = torch.load(model_path, map_location=device)
    enn.load_state_dict(saved['state'])
    opt.load_state_dict(saved['optimizer'])
    try:
        scheduler.load_state_dict(saved['scheduler'])
    except:
        scheduler = None
    history = saved['history']
    s0 = history[-1]['step'] + 1

else:
    history = []
    s0 = 0

# fit E3NN
for results in enn.fit(opt, dataloader_train, dataloader_valid, history, s0, max_iter=125, device=device,
                       scheduler=scheduler):
    with open(model_path, 'wb') as f:
        torch.save(results, f)

if not os.path.exists('images/' + enn.model_name + '_' + str(model_num)):
    os.makedirs('images/' + enn.model_name + '_' + str(model_num))

saved = torch.load(model_path, map_location=device)
history = saved['history']

steps = [d['step'] + 1 for d in history]
loss_train = [d['train']['loss'] for d in history]
loss_valid = [d['valid']['loss'] for d in history]

fig, ax = plt.subplots(figsize=(3.5,3))
ax.plot(steps, loss_train, label='Train', color=process.colors['Train'])
ax.plot(steps, loss_valid, label='Valid.', color=process.colors['Valid.'])

ax.set_xlabel('Iterations')
ax.set_ylabel('Loss')
ax.set_ylim(0,.1)
ax.set_title(element + '_XANES')
ax.legend(frameon=False)
ax.set_yscale('log')
fig.savefig('images/' + enn.model_name + '_' + str(model_num) + '/loss' + element + '.svg', bbox_inches='tight')

#peeking into the train and val loss curves
peeking_size = 25
print("The training and val loss look like:")
for i in range(peeking_size):
  print(loss_train[int(i*len(loss_train)/peeking_size)],loss_valid[int(i*len(loss_train)/peeking_size)])

#Comparing some
test_y_true = np.zeros((batch_size*len(dataloader_test),100))
test_y_pred = np.zeros((batch_size*len(dataloader_test),100))
current = 0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with torch.no_grad():
  for j,d in enumerate(dataloader_test):
    inputs = d.to(device)  # Adjust 'input' according to your data keys
    targets = d['y'].to(device)

    test_y_pred[current:current+len(d.y)] = enn(d).cpu().numpy()
    test_y_true[current:current+len(d.y)] = targets.cpu().numpy()
    current += len(d.y)

#Smoothing out the predictions through convolution
for i in range(len(test_y_pred)):
  temp = np.append(test_y_pred[i][0],test_y_pred[i])
  temp = np.append(temp,test_y_pred[i][len(test_y_pred[i])-1])
  temp = np.convolve(temp,[.25,.5,.25])
  temp = np.delete(temp,len(temp)-1)
  temp = np.delete(temp,len(temp)-1)
  temp = np.delete(temp,0)
  test_y_pred[i] = np.delete(temp,0)

#Getting the cosine similarity
#print('The Cosine similarity is')
#print(np.sum(test_y_pred*test_y_true)/((np.sum(test_y_pred**2))**(1/2)*(np.sum(test_y_true**2))**(1/2)))

#Find the average error:
#error = np.zeros(len(test_y_pred))
#for i in range(len(test_y_pred)):
  #error[i] = np.mean((test_y_pred[i]-test_y_true[i])**2)
  #if np.mean((test_y_pred[i]-test_y_true[i])**2) == 7.898989:
    #plt.plot(x_new,test_y_pred[i])
    #plt.plot(x_new,test_y_true[i])
#print('the element is ' + element)
#print("Quartile 1 is", np.percentile(error,25))
#print("The median is", np.median(error))
#print("Quartile 3 is", np.percentile(error,75))
#print(("The mean is"), np.mean(error))

#plt.hist(error,50,(0,np.percentile(error,99)))
#plt.title('Model Error Histogram for' + element)
#plt.ylabel('Count')
#plt.xlabel('Error')
#plt.savefig('images/' + element + '_Error.png', dpi=300)
#print("Average MSE of", np.mean(error))
#print("Median MSE of", np.median(error))

