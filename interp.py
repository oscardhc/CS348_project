import os
import PIL
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from r import *
from ress import DS
from torchvision.utils import make_grid, save_image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from utils import visualize_cam, explanation_map, Normalize
from gradcam import GradCAM, GradCAMpp


model_path = './reslinear/110.pth'
average_drop = 0
softmax_helper = nn.Softmax(dim=1)
original_conf = []
exp_conf = []
exp_pp_conf = []


cam_dict = dict()

resnet = Net()
resnet.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
resnet.eval()# , resnet.cuda(1)
# TODO: remove line 29 if use LSTM
resnet_model_dict = dict(model_type='Net', arch=resnet, layer_name='layer4_bottleneck0_bn3', input_size=(224, 224))

resnet_gradcam = GradCAM(resnet_model_dict, False)
resnet_gradcampp = GradCAMpp(resnet_model_dict, False)
cam_dict['resnet'] = [resnet_gradcam, resnet_gradcampp]
def visualize(torch_img, label, name): 

    normed_torch_img = torch_img

    torch_img_list = []
    for i in range(16): 
        torch_img_list.append(torch_img[i])

    original_result = softmax_helper(resnet(normed_torch_img))

    if not torch.argmax(original_result).item() == label: return False
    
    original_conf.append(original_result.view(-1)[label].item())

    '''-------------------- OK --------------------'''

    images = []
    for gradcam, gradcam_pp in cam_dict.values():
        mask, _ = gradcam(normed_torch_img)

        explain_map = []
        for i in range(16): 
            exp = explanation_map(mask[i], torch_img[i])
            explain_map.append(exp)
        
        # exp_map_tensor = mask * normed_torch_img
        # res = softmax_helper(resnet(exp_map_tensor))
        # exp_conf.append(res.view(-1)[label].item())

        # heatmap, result = [], []
        # for i in range(16): 
        #     h_i, r_i = visualize_cam(mask[i], torch_img[i])
        #     heatmap.append(h_i)
        #     result.append(r_i)
        # # heatmap, result = visualize_cam(mask, torch_img)

        mask_pp, _ = gradcam_pp(normed_torch_img)

        explain_map_pp = []
        for i in range(16): 
            exp = explanation_map(mask_pp[i], torch_img[i])
            explain_map_pp.append(exp)

        # exp_map_tensor = mask_pp * normed_torch_img
        # respp = softmax_helper(resnet(exp_map_tensor))
        # exp_pp_conf.append(respp.view(-1)[label].item())

        # heatmap_pp, result_pp = [], []
        # for i in range(16): 
        #     hpp_i, rpp_i = visualize_cam(mask_pp[i], torch_img[i])
        #     heatmap.append(hpp_i)
        #     result.append(rpp_i)
        # # heatmap_pp, result_pp = visualize_cam(mask_pp, torch_img)
        
        images.append(torch.stack(torch_img_list + explain_map + explain_map_pp, 0))
        
    images = make_grid(torch.cat(images, 0), nrow=16)

    output_dir = 'outputs'
    os.makedirs(output_dir, exist_ok=True)
    output_name = 'result' + str(name) + '.jpg'
    output_path = os.path.join(output_dir, output_name)

    save_image(images, output_path)

    # PIL.Image.open(output_path)
    return True

if __name__ == '__main__': 
    va = DataLoader(DS(False), batch_size=1, shuffle=False, num_workers=1)
    bar = tqdm(total=len(va))
    for i, (inp, lbl, tmp) in enumerate(va):
        # if i in [69, 74, 81, 90]:
        if i in [74]:
            print(tmp)
            inp = inp.reshape(16, 3, 224, 224)
            visualize(inp, lbl.item(), i)
            # if not visualize(inp, model_path, lbl.item(), i): continue
            bar.update(1)
        if i > 100: break
    bar.close()

    # ad = 0
    # ic = 0
    # ai = 0
    # adpp = icpp = aipp = 0
    # n = len(original_conf)

    # for o, g, gpp in zip(original_conf, exp_conf, exp_pp_conf): 
    #     ad += max(0, o - g) / o
    #     ai += max(0, g - o) / o
    #     ic += 1 if o < g else 0
    #     adpp += max(0, o - gpp) / o
    #     aipp += max(0, gpp - o) / o
    #     icpp += 1 if o < gpp else 0

    # ad /= n
    # ic /= n
    # ai /= n
    # adpp /= n
    # aipp /= n
    # icpp /= n
    # print(n, ad, ic, ai, adpp, icpp, aipp)
