#!/usr/bin/env python
from core import FlotDataset
from config import transformDefaultConfig
import torch
from nn.util import DataUtil
from tensorboardX import SummaryWriter
from PIL import Image
from PIL import ImageDraw

if __name__ == '__main__':
    #
    # The default configuration.
    conf = transformDefaultConfig.Config()
    train = FlotDataset.FlotDataset(conf, conf.dataTrainList, conf.transforms)
    dataset = torch.utils.data.DataLoader(train, batch_size = 1, num_workers = 1,
                                          shuffle = True, pin_memory = False)
    writer = SummaryWriter()
    for data in dataset:
        writer.add_image('Image', data['img'].cuda(), 0)
        writer.add_text('Text', 'text logged at step:'+str(1), 1)
        print(data['meta']['index'])
        print(data['meta']['shift'])
        print(data['labels'])
        name = '%sfront_camera_%d.png'%(data['meta']['filedir'][0],data['meta']['index'])
        print('consturcted name')
        print(name)
        img = Image.open(name)
        dx,dy = data['meta']['shift']
        w,h = img.size
        h_0 = int((h - conf.hyperparam.image_shape[0])/2 + dy)
        w_0 = int((w - conf.hyperparam.image_shape[1])/2 + dx)
        h_1 = h_0 + conf.hyperparam.image_shape[0]
        w_1 = w_0 + conf.hyperparam.image_shape[1]
        draw = ImageDraw.Draw(img)
        draw.rectangle([w_0, h_0, w_1, h_1])
        img.show()
        DataUtil.plotSample(data)
        # break
    writer.close()
