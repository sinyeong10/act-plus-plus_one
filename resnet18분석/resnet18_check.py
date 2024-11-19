import torch
import torch.nn as nn
import torchvision
from torchvision import models, transforms, utils
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
from PIL import Image
import json
# %matplotlib inline

transform = transforms.Compose([
    # transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=0., std=1.)
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

model = models.resnet18(pretrained=True)
print(model)

# we will save the conv layer weights in this list
model_weights =[]
#we will save the 49 conv layers in this list
conv_layers = []
# get all the model children as list
model_children = list(model.children())
#counter to keep count of the conv layers
counter = 0
#append all the conv layers and their respective wights to the list
for i in range(len(model_children)):
    if type(model_children[i]) == nn.Conv2d:
        counter+=1
        model_weights.append(model_children[i].weight)
        conv_layers.append(model_children[i])
    elif type(model_children[i]) == nn.Sequential:
        for j in range(len(model_children[i])):
            for child in model_children[i][j].children():
                if type(child) == nn.Conv2d:
                    counter+=1
                    model_weights.append(child.weight)
                    conv_layers.append(child)
print(f"Total convolution layers: {counter}")
print("conv_layers")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

for idx in range(9,10):
    image = Image.open(str(f'scr\\mask_data\\two_cam_episode_46_image\\top_frame_{idx}.jpg'))
    plt.imshow(image)

    image = transform(image)
    print(f"Image shape before: {image.shape}")
    image = image.unsqueeze(0)
    print(f"Image shape after: {image.shape}")
    image = image.to(device)

    outputs = []
    names = []
    for layer in conv_layers[0:]:
        image = layer(image)
        outputs.append(image)
        names.append(str(layer))
    print(len(outputs)) #17개
    #print feature_maps
    for feature_map in outputs:
        print(feature_map.shape)




    processed = []
    for feature_map in outputs:
        feature_map = feature_map.squeeze(0)
        gray_scale = torch.sum(feature_map,0)
        gray_scale = gray_scale / feature_map.shape[0]
        processed.append(gray_scale.data.cpu().numpy())
    for fm in processed:
        print(fm.shape)

#[1,3,480,640]

# torch.Size([1, 64, 240, 320])
# torch.Size([1, 64, 240, 320])
# torch.Size([1, 64, 240, 320])
# torch.Size([1, 64, 240, 320])
# torch.Size([1, 64, 240, 320])
# torch.Size([1, 128, 120, 160])
# torch.Size([1, 128, 120, 160])
# torch.Size([1, 128, 120, 160])
# torch.Size([1, 128, 120, 160])
# torch.Size([1, 256, 60, 80])
# torch.Size([1, 256, 60, 80])
# torch.Size([1, 256, 60, 80])
# torch.Size([1, 256, 60, 80])
# torch.Size([1, 512, 30, 40])
# torch.Size([1, 512, 30, 40])
# torch.Size([1, 512, 30, 40])
# torch.Size([1, 512, 30, 40])
# (240, 320)
# (240, 320)
# (240, 320)
# (240, 320)
# (240, 320)
# (120, 160)
# (120, 160)
# (120, 160)
# (120, 160)
# (60, 80)
# (60, 80)
# (60, 80)
# (60, 80)
# (30, 40)
# (30, 40)
# (30, 40)
# (30, 40)



    fig = plt.figure(figsize=(30, 50))
    for i in range(len(processed)):
        a = fig.add_subplot(5, 4, i+1)
        imgplot = plt.imshow(processed[i])
        a.axis("off")
        a.set_title(names[i].split('(')[0], fontsize=10)
    plt.savefig(str('feature_maps.jpg'), bbox_inches='tight')

    plt.show()