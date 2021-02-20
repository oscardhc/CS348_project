
from __future__ import division
from __future__ import with_statement
from __future__ import print_function
from collections import OrderedDict
import torch
import torch.nn as nn
import os
import torch.optim as optim
from torch.utils import model_zoo
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from rlstm import *

np.random.seed(1234)

c = pd.read_csv('out.csv')
spl = set(np.random.permutation(len(c.index))[:len(c.index) // 5])

for i in range(100):
    print('*' if i in spl else ' ', end='')

class DS(Dataset):
    
    def __init__(self, trainSet):
        global c, spl
        self.imgs = []
        self.lbls = []
        self.cont = []
        print('LOADING IMAGES...')
        bar = tqdm(total = len(c.index))
        for idx, val in c.iterrows():
            bar.update(1)
            if trainSet != (idx in spl):
                if val[2] < 16:
                    continue
                ccc = min(val[2], 32)
                bar.set_description(f'count={ccc}')
                self.cont.append(ccc)
                self.imgs.append((idx, val[2]))
                self.lbls.append(val[1] - 1)
        bar.close()

    def __getitem__(self, index):
        tup = self.imgs[index]
        ccc = self.cont[index]
        arr = []
        for i in range(0, tup[1], tup[1] // ccc):
            img = Image.open(f'dt/{tup[0]}_{i}.jpg')
            dat = np.asarray(img).astype(np.uint8).transpose((2, 0, 1))
            arr.append(dat)
            if len(arr) == ccc:
                break
        arr = np.stack(arr)
        rd = np.sort(np.random.permutation(self.cont[index])[:16])
        return np.stack(arr[rd]).astype(np.single) / 255, self.lbls[index]

    def __len__(self):
        return len(self.imgs) // 64 * 64


if __name__ == '__main__':

    dv = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = Net().to(dv)
    bs = 4

    # model.load_state_dict(torch.load('100.pth'))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    dl = DataLoader(DS(True), batch_size=bs, shuffle=True, num_workers=16, pin_memory=True)
    va = DataLoader(DS(False), batch_size=bs, shuffle=True, num_workers=16, pin_memory=True)
    print('data_train: ', len(dl), '*', bs)
    print('data_val  : ', len(va), '*', bs)

    allloss = []
    allaccr = []
    tstaccr = []

    hd = (torch.randn(1, bs, 83).to(dv), torch.randn(1, bs, 83).to(dv))

    for ep in range(1001):
        eploss = []
        epaccr = []
        if True:
            bar = tqdm(total = len(dl) * bs)
            for idx, it in enumerate(dl):
                inp, lbl = it
                inp = inp.to(dv)
                inp = inp.reshape(bs * 16, 3, 224, 224)

                optimizer.zero_grad()
                out = model(inp).cpu()
                loss = criterion(out, lbl)
                loss.backward()
                optimizer.step()

                epaccr.append((sum(torch.max(out, 1)[1] == lbl) / bs).item())
                eploss.append(loss.item())
                bar.set_description(f'loss = {loss.item()}')
                bar.update(bs)
            allloss.append(np.array(eploss).mean())
            allaccr.append(np.array(epaccr).mean())
            bar.set_description(f'epoch %d loss=%.4f acc=%.3f' % (ep, allloss[-1], allaccr[-1]))
            bar.close()
        if ep % 2 == 0:
            with torch.no_grad():
                accr = []
                for inp, lbl in va:
                    inp = inp.reshape(bs * 16, 3, 224, 224)
                    out = model(inp.to(dv)).cpu()
                    accr.append(sum(torch.max(out, 1)[1] == lbl) / bs)
                tstaccr.append(np.array(accr).mean())
            print('testing accuracy =', tstaccr[-1])
        if ep % 10 == 0:
            torch.save(model.state_dict(), f'./models4/{ep}.pth')
            with open('res4.txt', 'w') as f:
                print(allloss, file=f)
                print(allaccr, file=f)
                print(tstaccr, file=f)
