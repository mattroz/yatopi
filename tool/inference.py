import os
import torch
from argparse import  ArgumentParser
import torch.nn as nn
from torch.utils.data import DataLoader
from data_generator import ImageDataLoader, randomSequentialStratifiedSampler, randomSequentialSampler, batchOvertfitSampler
import matplotlib.pyplot as plt
import numpy as np

import mlflow

from tqdm import tqdm
import transforms
from model import SampleClassificationNetwork, init_weights
from arguments import argp

argp = ArgumentParser()
argp.add_argument('--dir', type=str, required=True)
argp.add_argument('--weights', type=str, required=True)
argp = argp.parse_args()

cuda_flag = False

model = SampleClassificationNetwork(nclasses=4)

if torch.cuda.is_available():
    cuda_flag = True
    state_dict = torch.load(argp.weights, map_location='cuda')
    model.load_state_dict(state_dict['model_state_dict'])
    model = model.to(torch.device('cuda'))
else:
    print("CPU")
    state_dict = torch.load(argp.weights, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict['model_state_dict'])

model.eval()


transformer = transforms.ResizeNormalize((224, 224), imagenet=True)


_dir = argp.dir.split('/')[-2]
if not os.path.isdir(os.path.join('./predictions', _dir)):
    os.makedirs(os.path.join('./predictions', _dir))

for file in tqdm(os.listdir(argp.dir)):
    timestamp = file.split('_')[0]
    if not os.path.splitext(file)[-1] in ['.png', '.jpeg', '.jpg', '.gif']:
        continue

    img_path = os.path.join(argp.dir, file)
    #target = img_path.split('_')[-2]

    image_orig = Image.open(img_path).convert('L')
    image = cv2.cvtColor(np.array(image_orig), cv2.COLOR_GRAY2RGB)
    image = transformer(image)['image']

    #cv2.imshow("", image.numpy()[0])
    #cv2.waitKey()

    if cuda_flag:
        image = image.cuda()
    image = image.view(1, *image.size())

    with torch.no_grad():
        output = model(image)
        #print(output.softmax(1))#.argmax(dim=1))
        pred = decode_output(output.argmax(dim=1), ALPHABET)
        print(pred)
        pred = re.sub('[$]', '', pred)


    cv2.imwrite(f'./predictions/{_dir}/{timestamp}_{pred}.jpg', np.float32(image_orig))

    # cv2.imshow("demo", demo)
    # cv2.waitKey()
