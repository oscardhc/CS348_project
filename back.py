""" How to use C3D network. """
import numpy as np

import torch

from os.path import join
from glob import glob

import skimage.io as io
from skimage.transform import resize

from C3D_model import C3D
from ress import DS

from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

def get_sport_clip(clip_name, verbose=True):
    """
    Loads a clip to be fed to C3D for classification.
    TODO: should I remove mean here?
    
    Parameters
    ----------
    clip_name: str
        the name of the clip (subfolder in 'data').
    verbose: bool
        if True, shows the unrolled clip (default is True).

    Returns
    -------
    Tensor
        a pytorch batch (n, ch, fr, h, w).
    """

    clip = sorted(glob(join('data', clip_name, '*.png')))
    clip = np.array([resize(io.imread(frame), output_shape=(112, 200), preserve_range=True) for frame in clip])
    clip = clip[:, :, 44:44+112, :]  # crop centrally

    if verbose:
        clip_img = np.reshape(clip.transpose(1, 0, 2, 3), (112, 16 * 112, 3))
        io.imshow(clip_img.astype(np.uint8))
        io.show()

    clip = clip.transpose(3, 0, 1, 2)  # ch, fr, h, w
    clip = np.expand_dims(clip, axis=0)  # batch axis
    clip = np.float32(clip)

    return torch.from_numpy(clip)



def main():
    """
    Main function.
    """

    # get network pretrained model
    net = C3D()

    d = torch.load('c3d.pickle')
    for k, v in d.items():
        if k == 'fc8.weight' or k == 'fc8.bias':
            print(d[k].size())
            d[k] = v[:83,].normal_(mean=0, std=0.5)
            print(d[k].size())
    net.load_state_dict(d)

    dv = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(dv)
    net = net.to(dv)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-5)

    bs = 16
    dl = DataLoader(DS(True), batch_size=bs, shuffle=True, num_workers=16, pin_memory=True)
    va = DataLoader(DS(False), batch_size=bs, shuffle=True, num_workers=16, pin_memory=True)

    print('data_train: ', len(dl), '*', bs)
    print('data_val  : ', len(va), '*', bs)

    allloss = []
    allaccr = []
    tstaccr = []

    for ep in range(1001):
        eploss = []
        epaccr = []
        bar = tqdm(total = len(dl) * bs)
        for idx, it in enumerate(dl):
            inp, lbl = it
            inp.transpose_(2, 1)
            inp = inp.to(dv)

            optimizer.zero_grad()
            out = net(inp).cpu()
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
                    out = net(inp.to(dv)).cpu()
                    accr.append(sum(torch.max(out, 1)[1] == lbl) / bs)
                tstaccr.append(np.array(accr).mean())
            print('testing accuracy =', tstaccr[-1])
        if ep % 10 == 0:
            torch.save(net.state_dict(), f'./models3/{ep}.pth')
            with open('res3.txt', 'w') as f:
                print(allloss, file=f)
                print(allaccr, file=f)
                print(tstaccr, file=f)

# entry point
if __name__ == '__main__':
    main()
