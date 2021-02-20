import os
import PIL
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from C3D_model import *
from ress import DS
from torchvision.utils import make_grid, save_image
from torch.utils.data import Dataset, DataLoader

from utils import visualize_cam, explanation_map, Normalize
from gradcam import GradCAM, GradCAMpp

model_path = './models/80.pth'
def visualize(torch_img, lbl, name): 

    normed_torch_img = torch_img.permute(0, 2, 1, 3, 4)
    torch_img = torch_img[0]

    torch_img_list = []
    for i in range(16): 
        torch_img_list.append(torch_img[i])

    PATH = model_path

    C3D_net = C3D()
    C3D_net.load_state_dict(torch.load(PATH, map_location=lambda storage, loc: storage))
    C3D_net.eval()# , C3D.cuda(1)

    # if not torch.argmax(C3D_net(normed_torch_img)).item() == lbl: return False

    cam_dict = dict()

    C3D_model_dict = dict(model_type='C3D', arch=C3D_net, layer_name='last', input_size=(16, 224, 224))
    '''-------------------- OK --------------------'''
    C3D_gradcam = GradCAM(C3D_model_dict, False)
    C3D_gradcampp = GradCAMpp(C3D_model_dict, False)
    cam_dict['C3D'] = [C3D_gradcam, C3D_gradcampp]

    images = []

    mask, _ = C3D_gradcam(normed_torch_img)
    mask = mask.permute(0, 2, 1, 3, 4)
    mask = mask[0]

    explain_map = []
    for i in range(16): 
        exp = explanation_map(mask[i], torch_img[i])
        explain_map.append(exp)
    # heatmap, result = [], []
    # for i in range(16): 
    #     h_i, r_i = visualize_cam(mask[i], torch_img[i])
    #     heatmap.append(h_i)
    #     result.append(r_i)
    # heatmap, result = visualize_cam(mask, torch_img)

    mask_pp, _ = C3D_gradcampp(normed_torch_img)
    mask_pp = mask_pp.permute(0, 2, 1, 3, 4)
    mask_pp = mask_pp[0]

    explain_pp_map = []
    for i in range(16): 
        exp = explanation_map(mask_pp[i], torch_img[i])
        explain_pp_map.append(exp)
    # heatmap_pp, result_pp = [], []
    # for i in range(16): 
    #     hpp_i, rpp_i = visualize_cam(mask_pp[i], torch_img[i])
    #     heatmap_pp.append(hpp_i)
    #     result_pp.append(rpp_i)
    # heatmap_pp, result_pp = visualize_cam(mask_pp, torch_img)
    
    # images.append(torch.stack(torch_img_list + explain_map + explain_pp_map, 0))
        
    # images = make_grid(torch.cat(images, 0), nrow=16)


    # output_dir = './outputs'
    # os.makedirs(output_dir, exist_ok=True)
    # output_name = 'grad_cam_result.jpg'
    # output_path = os.path.join(output_dir, output_name)

    # save_image(images, output_path)
    # PIL.Image.open(output_path)
    images = []
    images.append(torch.stack(torch_img_list + explain_map + explain_pp_map, 0))

    images = make_grid(torch.cat(images, 0), nrow=16)


    output_dir = './outputs'
    os.makedirs(output_dir, exist_ok=True)
    output_name = 'result' + str(name) + '.jpg'
    output_path = os.path.join(output_dir, output_name)

    save_image(images, output_path)

    return True

if __name__ == '__main__': 
    va = DataLoader(DS(False), batch_size=1, shuffle=False)
    for i, (inp, lbl) in enumerate(va):
        if i > 80: break
        if not i == 74: continue
        print(i)
        visualize(inp, lbl.item(), i)