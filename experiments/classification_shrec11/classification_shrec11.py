import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../../src/"))  
import fannet
from shrec11_dataset import Shrec11MeshDataset_Simplified, Shrec11MeshDataset_Original

def list_of_floats(arg):
    return list(map(float,arg.split(',')))

# === Options

# Parse a few args
parser = argparse.ArgumentParser()
parser.add_argument("--input_features", type=str, help="what features to use as input ('xyz' or 'hks') default: hks", default = 'hks')
parser.add_argument("--dataset_type", type=str, help="which variant of the dataset to use ('original', or 'simplified') default: original", default = 'original')
parser.add_argument("--split_size", type=int, help="how large of a training set per-class default: 10", default=10)
parser.add_argument("--spoke_length", type=list_of_floats, default='0.00,0.01,0.02,0.03')
args = parser.parse_args()

# system things
device = torch.device('cuda:0')
dtype = torch.float32

# problem/dataset things
n_class = 30

# model 
input_features = args.input_features # one of ['xyz', 'hks']
k_eig = 128

# training settings
n_epoch = 200
lr = 1e-4
decay_every = 9999
decay_rate = 0.5
augment_random_rotate = (input_features == 'xyz')
label_smoothing_fac = 0.2

spoke_length = args.spoke_length

spoke_dir = os.path.join('op_cache_' + '_'.join(str(e) for e in spoke_length) )

# Important paths
base_path = os.path.dirname(__file__)
op_cache_dir = os.path.join(base_path, "data", spoke_dir)

if args.dataset_type == "simplified":
    dataset_path = os.path.join(base_path, "data/simplified")
elif args.dataset_type == "original":
    dataset_path = os.path.join(base_path, "data/original")
else:
    raise ValueError("Unrecognized dataset type")


# === Load datasets

#spoke_length=[0.00,0.01,0.02,0.03]

# Train dataset
if args.dataset_type == "simplified":
    train_dataset = Shrec11MeshDataset_Simplified(dataset_path, split_size=args.split_size,
                                                  k_eig=k_eig, op_cache_dir=op_cache_dir, spoke_length=spoke_length)
elif args.dataset_type == "original":
    train_dataset = Shrec11MeshDataset_Original(dataset_path, split_size=args.split_size,
                                                  k_eig=k_eig, op_cache_dir=op_cache_dir, spoke_length=spoke_length)
train_loader = DataLoader(train_dataset, batch_size=None, shuffle=True)

# Test dataset
if args.dataset_type == "simplified":
    test_dataset = Shrec11MeshDataset_Simplified(dataset_path, split_size=None,
                                                 k_eig=k_eig, op_cache_dir=op_cache_dir,
                                                 spoke_length=spoke_length,
                                                 exclude_dict=train_dataset.entries)
elif args.dataset_type == "original":
    test_dataset = Shrec11MeshDataset_Original(dataset_path, split_size=None,
                                                 k_eig=k_eig, op_cache_dir=op_cache_dir,
                                                 spoke_length=spoke_length,
                                                 exclude_dict=train_dataset.entries)
test_loader = DataLoader(test_dataset, batch_size=None)





# === Create the model

C_in={'xyz':3, 'hks':16}[input_features] # dimension of input features

                                     
model = fannet.layers_classification.Net(in_channels=C_in, num_classes=n_class, seq_length=6 * len(spoke_length) + 1)

model = model.to(device)

print(model)

# === Optimize
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

def train_epoch(epoch):

    # Implement lr decay
    if epoch > 0 and epoch % decay_every == 0:
        global lr 
        lr *= decay_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr 


    print(f'Epoch: {epoch}, LR = {optimizer.param_groups[0]["lr"]}')

    # Set model to 'train' mode
    model.train()
    optimizer.zero_grad()
    
    correct = 0
    total_num = 0
    for data in tqdm(train_loader):

        # Get data
        verts, faces, frames, mass, L, evals, evecs, gradX, gradY, labels, fanIndices = data

        # Move to device
        verts = verts.to(device)
        faces = faces.to(device)
        frames = frames.to(device)
        mass = mass.to(device)
        L = L.to(device)
        evals = evals.to(device)
        evecs = evecs.to(device)
        gradX = gradX.to(device)
        gradY = gradY.to(device)
        labels = labels.to(device)
        fanIndices = fanIndices.to(device)
        
        # Randomly rotate positions
        if augment_random_rotate:
            verts = fannet.utils.random_rotate_points(verts)

        # Construct features
        if input_features == 'xyz':
            features = verts
        elif input_features == 'hks':
            features = fannet.geometry.compute_hks_autoscale(evals, evecs, 16)
        
        preds = model(features, fanIndices, mass)

        # Evaluate loss
        loss = fannet.utils.label_smoothing_log_loss(preds, labels, label_smoothing_fac)
        loss.backward()
        
        # track accuracy
        pred_labels = torch.max(preds, dim=-1).indices
        this_correct = pred_labels.eq(labels).sum().item()
        correct += this_correct
        total_num += 1

        # Step the optimizer
        optimizer.step()
        optimizer.zero_grad()

    train_acc = correct / total_num
    return train_acc

# Do an evaluation pass on the test dataset 
def test():
    
    model.eval()
    
    correct = 0
    total_num = 0
    with torch.no_grad():
    
        for data in tqdm(test_loader):

            # Get data
            verts, faces, frames, mass, L, evals, evecs, gradX, gradY, labels, fanIndices = data

            # Move to device
            verts = verts.to(device)
            faces = faces.to(device)
            frames = frames.to(device)
            mass = mass.to(device)
            L = L.to(device)
            evals = evals.to(device)
            evecs = evecs.to(device)
            gradX = gradX.to(device)
            gradY = gradY.to(device)
            labels = labels.to(device)
            fanIndices = fanIndices.to(device)
            
            # Construct features
            if input_features == 'xyz':
                features = verts
            elif input_features == 'hks':
                features = fannet.geometry.compute_hks_autoscale(evals, evecs, 16)

            # Apply the model
            preds = model(features, fanIndices, mass)

            # track accuracy
            pred_labels = torch.max(preds, dim=-1).indices
            this_correct = pred_labels.eq(labels).sum().item()
            correct += this_correct
            total_num += 1

    test_acc = correct / total_num
    return test_acc 


print("Training...")


best_acc_epoch = {
    'epoch': 0,
    'test_acc': 0.00
}

for epoch in range(n_epoch):
    train_acc = train_epoch(epoch)
    test_acc = test()
    if test_acc > best_acc_epoch['test_acc']:
        best_acc_epoch['test_acc'] = test_acc
        best_acc_epoch['epoch'] = epoch
    print("Epoch {} - Train overall: {:06.3f}%  Test overall: {:06.3f}%".format(epoch, 100*train_acc, 100*test_acc))

# Test
test_acc = test()
print("Overall test accuracy: {:06.3f}%".format(100*test_acc))

print('##########################')
print(f"Best Accuracy Value {best_acc_epoch['test_acc']} , Epoch: {best_acc_epoch['epoch']}")
