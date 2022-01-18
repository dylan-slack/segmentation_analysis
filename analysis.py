"""
This code is adapted from code here:
https://github.com/marcotcr/lime/blob/master/doc/notebooks/Tutorial%20-%20images%20-%20Pytorch.ipynb
"""
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn as nn
import numpy as np
import os, json

import torch
from torchvision import models, transforms
from torch.autograd import Variable
import torch.nn.functional as F

from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
from tqdm import tqdm
from skimage.segmentation import mark_boundaries
import pandas as pd

np.random.seed(0)

def get_image(path):
    # Takne from https://github.com/marcotcr/lime/blob/master/doc/notebooks/Tutorial%20-%20images%20-%20Pytorch.ipynb
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB') 


def get_input_transform():
    # Taken from https://github.com/marcotcr/lime/blob/master/doc/notebooks/Tutorial%20-%20images%20-%20Pytorch.ipynb
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])       
    transf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])    

    return transf


def get_input_tensors(imgs):
    # modified from https://github.com/marcotcr/lime/blob/master/doc/notebooks/Tutorial%20-%20images%20-%20Pytorch.ipynb
    transf = get_input_transform()
    return transf(imgs)


def get_inception():
    return models.inception_v3(pretrained=True)


def model_iterator():
    models = [get_inception]
    for model in models:
        model_ = model()
        model_.eval()
        yield model_.cuda()
        del model_


def get_image_files(rootdir):
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            if file.endswith('.jpg'):
                yield os.path.join(rootdir, file)


def process_images(image_files):
    # modified from https://github.com/marcotcr/lime/blob/master/doc/notebooks/Tutorial%20-%20images%20-%20Pytorch.ipynb
    images, imags_pil = [], []
    for f in image_files:
        img = get_image(f)
        imags_pil.append(img)
        images.append(get_input_tensors(img))
    return torch.stack(images, dim=0), imags_pil


def get_pil_transform(): 
    # modified from https://github.com/marcotcr/lime/blob/master/doc/notebooks/Tutorial%20-%20images%20-%20Pytorch.ipynb
    transf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224)
    ])    

    return transf


def get_preprocess_transform():
    # modified from https://github.com/marcotcr/lime/blob/master/doc/notebooks/Tutorial%20-%20images%20-%20Pytorch.ipynb
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])     
    transf = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])    

    return transf


def yield_segmentation_f(seg_name):
    if seg_name == 'quickshift':
        for r in [0.01, 0.2]:
            for k_size in [3, 8]:
                for max_dist in [100, 1000]:
                    segmentation_fn = SegmentationAlgorithm(seg_name, kernel_size=k_size,
                                                                max_dist=max_dist, ratio=r,
                                                                random_seed=0)

                    params = {
                        'name': seg_name,
                        'ratio': r,
                        'k_size': k_size,
                        'max_dist': max_dist
                    }

                    yield segmentation_fn, params

    elif seg_name == 'slic':
        for n_segs in [10, 40]:
            for c in [0.1, 1, 10]:
                segmentation_fn = SegmentationAlgorithm(seg_name, n_segments=n_segs,
                                                                compactness=c,
                                                                random_seed=0)
                params = {
                    'name': seg_name,
                    'compactness': c,
                    'n_segments': n_segs
                }
                yield segmentation_fn, params

    elif seg_name == 'felzenszwalb':
        for scale in [10, 25]:
            for sigma in [10, 100]:
                segmentation_fn = SegmentationAlgorithm(seg_name, scale=scale, sigma=sigma,
                                                                random_seed=0)
                params = {
                    'name': seg_name,
                    'sigma': sigma,
                    'scale': scale
                }
                yield segmentation_fn, params


def add(d, key, value):
    if key not in d:
        d[key] = [value]
    else:
        d[key].append(value)

def main():

    files = [f for f in get_image_files('./data')]
    data, data_pil = process_images(files)
    print(f'Data has shape {data.shape}')

    pill_transf = get_pil_transform()
    preprocess_transform = get_preprocess_transform()

    for model in model_iterator():

        def batch_predict(images):
            model.eval()
            batch = torch.stack(tuple(preprocess_transform(i) for i in images), dim=0)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            batch = batch.to(device)
            
            logits = model(batch)
            probs = F.softmax(logits, dim=1)
            return probs.detach().cpu().numpy()

        data = {}

        segment_lens = {
            'quickshift': 2 * 2 * 2,
            'slic': 2 * 3,
            'felzenszwalb': 2 * 3

        }

        for key in segment_lens:
            data[key] = {}

        for seg_name in tqdm(segment_lens):
            for img_num, img in enumerate(tqdm(data_pil[:100])):
                for sf, params in tqdm(yield_segmentation_f(seg_name), total=segment_lens[seg_name]):
                    explainer = lime_image.LimeImageExplainer()
                    explanation = explainer.explain_instance(np.array(pill_transf(img)), 
                                                 batch_predict,
                                                 segmentation_fn=sf,
                                                 top_labels=1, 
                                                 hide_color=0, 
                                                 num_samples=1_000)

                    top_label = explanation.top_labels[0]
                    num_sp = len(explanation.local_exp[top_label])

                    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=num_sp, hide_rest=False)
                    img_boundry1 = mark_boundaries(np.array(pill_transf(img))/255.0, explanation.segments)

                    fname = ''
                    for key in params:
                        fname += key
                        fname += '_'
                        fname += str(params[key])
                        fname += '_'
                        add(data[params['name']], key, params[key])

                    fname += '.jpg'
                    ndir = './results/img_' + str(img_num)
                    if not os.path.exists(ndir):
                        os.mkdir('./results/img_' + str(img_num))
                    plt.imsave(f'results/img_{str(img_num)}/{fname}', img_boundry1)

                    fidelity = explanation.score

                    add(data[params['name']], 'fidelity', fidelity)
                    add(data[params['name']], 'n_segs', num_sp)
                    add(data[params['name']], 'img_num', img_num)


            df = pd.DataFrame(data=data[seg_name])
            df.to_csv(f'{seg_name}.csv')
  

if __name__ == "__main__":
    main()