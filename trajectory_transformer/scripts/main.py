import os
from random import sample
import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


from trajectory_transformer.model import Transformer,subsequent_mask
from trajectory_transformer.data import datapath
from trajectory_transformer.scripts import trajectory_dataset
from trajectory_transformer.scripts import train_loop,test_loop,save_outputs
from trajectory_transformer.log import logpath

data_dir = datapath+"/double_pendulum/0102_1327"
fnames = []
for root, dirs, files in os.walk(data_dir):
    for f in files:
        fnames.append(os.path.join(root,f))

test_percent = 0.01
n_files = len(fnames)
n_train = n_files - int(test_percent*n_files)
n_test = int(test_percent*n_files)

train_files = sample(fnames,n_train)
test_files = [f for f in fnames if f not in train_files]

data_len = 10
pred_len = 10
train_data = trajectory_dataset(train_files,data_len,pred_len)
test_data = trajectory_dataset(test_files,data_len,pred_len)

train_dataloader = DataLoader(train_data, batch_size=100, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=100, shuffle=True)

save_dir = logpath + "/{}".format(datetime.datetime.now().strftime('%m%d_%H%M'))
os.mkdir(save_dir)

layers=2
d_model=512
d_ff=2048
heads=8
dropout=0.1
device = "mps"
model = Transformer(layers, d_model, d_ff, heads, dropout).to(device)

src_mask = (torch.ones(1,data_len,data_len) == 1).to(device)
tgt_mask = subsequent_mask(pred_len).to(device)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.98), eps=1e-9)

epochs = 1
loss,test_loss = [],[]
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    loss += train_loop(train_dataloader, model, src_mask, tgt_mask, loss_fn, optimizer, device)
    test_loss += test_loop(test_dataloader, model, src_mask, tgt_mask, loss_fn, device)
    save_outputs(t,loss,test_loss,model,save_dir)
print("Done!")

torch.save(model.state_dict(),save_dir+"/model.pt")

