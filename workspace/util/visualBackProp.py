import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models,transforms
import os
from skimage import io 
from data_loader import imageandlabel
import torchvision.utils as vutils
import pdb

import cv2

from scipy.misc import imsave


#Since pytorch does not save intermediate outputs unlike torch/lua, 
#a new class is created with a newly defined forward function

def normalization(tensor):
    omin = tensor.min(2,keepdim=True)[0].min(3,keepdim=True)[0].mul(-1)
    omax = tensor.max(2,keepdim=True)[0].max(3,keepdim=True)[0].add(omin)
    tensor = torch.add(tensor,omin.expand(tensor.size(0), tensor.size(1), tensor.size(2), tensor.size(3)))
    tensor = torch.div(tensor,omax.expand(tensor.size(0),tensor.size(1), tensor.size(2), tensor.size(3)))
    return tensor

def rtobTable():
    rng = 256
    r,g,b = 0, 20, 255
    dr = 1.
    dg = 1.
    db = -5
    rgb = []
    for i in range(rng):
        r,g,b = r+dr, g+dg, b+db
        if r > 255:
            r = 255
            dr = 0
        elif r < 0:
            r = 0
            dr = 0

        if g > 255:
            g = 255
            dg = 0
        elif g < 0:
            g = 0
            dg = 0

        if b > 255:
            b = 255
            db = 0
        if b < 0:
            b = 0
            db = 0

        if i>15:
            rgb.append((int(r), int(g), int(b)))
        else:
            rgb.append((0,0,0))
    # return np.array(rgb)
    return torch.LongTensor(rgb)

#myFeatureExtractor
from torchvision.models import resnet

class myFeatureExtractor(resnet.ResNet):

    def __init__(self,model):
        super(myFeatureExtractor, self).__init__(resnet.BasicBlock, [2, 2, 2, 2])
        self = model.model

    def handlesequential(self,x,module,output):
        i = len(output)

        for name,submodule in module._modules.items():
            x = submodule(x)
            output[i] = x.data.clone()
            i+=1

        return x,output 

    def forward(self, x):
        outputs = {}
        i=0
        #iterating over modules in the model
        for name,submodule in self._modules.items():
            if name.find('relu') != -1 :
                x = submodule(x)
                outputs[i] = x.data.clone()
                i+=1

            elif name.find('layer') != -1:
                x,outputs = self.handlesequential(x,submodule,outputs)
                i = len(outputs)

            else:
                if name.find('fc') != -1:
                    x = x.view(x.size(0),-1)
                x = submodule(x)

        return outputs


class VisualBackProp(object):
    def __init__(self, model):
        # self.model = model
        model = myFeatureExtractor(model)
        if torch.cuda.is_available():
            model = model.cuda()
        model.eval()
        model.train(False)
        self.model = model
        self.rgbtable = rtobTable()

    # name='24-02-2018-model_best.pth.tar'
    # model = myFeatureExtractor(name)

    def vismask_res(self, img):
        output = self.model(img)

        summation = {}
        sumUp = {}

        for i in range(len(output)-1,-1,-1):
            #sum all feature maps in a lyer
            summation[i] = output[i].sum(1,keepdim=True)


            #point wise multiplication (multiplying output with the previous layer (backpropagating))
            if i < len(output)-1:
                summation[i] = torch.mul(summation[i],sumUp[i + 1])
                summation[i] = normalization(summation[i])

                # summation[i][summation[i] > 0.25] = 1

            #save the intermediate mask (image obtained by backpropagating at every layer)

            if i > 0:
                if output[i].size() == output[i-1].size():

                    #scaling up the feature map using deconvolution operation
                    mmUp = nn.ConvTranspose2d(1,1,kernel_size=(3,3),stride=(1,1),padding=(1,1))
                    mmUp.weight.data.fill_(1)
                    mmUp.bias.data.fill_(0)

                    if torch.cuda.is_available():
                        mmUp.cuda()
                        sumUp[i] = mmUp(Variable(summation[i].cuda(), volatile=True)).data.clone()
                    else:
                        sumUp[i] = mmUp(Variable(summation[i], volatile=True)).data.clone()

                else:

                    mmUp = nn.ConvTranspose2d(1,1,kernel_size=(6,6),stride=(2,2),padding=(2,2))

                    if torch.cuda.is_available():
                        mmUp.cuda()

                    mmUp.weight.data.fill_(1)
                    mmUp.bias.data.fill_(0)

                    if torch.cuda.is_available():
                        sumUp[i] = mmUp(Variable(summation[i].cuda(),volatile=True)).data.clone()
                    else:
                        sumUp[i] = mmUp(Variable(summation[i],volatile=True)).data.clone()

            else:
                mmUp = nn.ConvTranspose2d(1,1,kernel_size=(7,7),stride=(2,2),padding=(3,3),output_padding=(1,1))

                if torch.cuda.is_available():
                    mmUp.cuda()

                mmUp.weight.data.fill_(1)
                mmUp.bias.data.fill_(0)

                if torch.cuda.is_available():
                    sumUp[i] = mmUp(Variable(summation[i].cuda(),volatile=True)).data.clone()
                else:
                    sumUp[i] = mmUp(Variable(summation[i],volatile=True)).data.clone()


        #normalizing the final mask.
        out = sumUp[i]
        out = normalization(out)

        return out

    def visualize(self, imgRaw):

        i=0
        vismask = self.vismask_res(imgRaw)
        img = imgRaw

        img[i,0].data.mul_(0.229).add_(0.485)
        img[i,1].data.mul_(0.224).add_(0.456)
        img[i,2].data.mul_(0.225).add_(0.406)

        img = img[i].data*255.0
        mask = vismask[i]
        mask = mask*255.0/torch.max(mask)
        mask = mask.type(torch.LongTensor)

        d, w, h = mask.size()
        mask = mask.view(mask.numel())
        mask = mask.unsqueeze(1)

        ret = torch.LongTensor(w*h, 3).zero_()
        ret[:,:] = self.rgbtable[mask,:]
        colored_mask = ret.view(w,h,d*3).type(torch.FloatTensor)

        img = img.transpose(0, 2).transpose(0,1).type(torch.FloatTensor)

        out = torch.add(img, 0.4, colored_mask)
        return out
        # imsave(save + path[i].split('.')[0] + str('final') + '.png', out)

# Scaling and normalizing the images to required sizes (mean and std deviation are values required by trained VGG model)
# trans = transforms.Compose([transforms.Resize(400),
    # transforms.CenterCrop(224),
    # transforms.ToTensor(),
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

#------------------------------------------------------------------------------------

