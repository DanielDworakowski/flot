#!/usr/bin/env python
import FlotDataset
import DefaultNNConfig
import torch
from DataUtil import plotSample
from tensorboardX import SummaryWriter
from torch.autograd import Variable

if __name__ == '__main__':
    #
    # The default configuration.
    conf = DefaultNNConfig.Config()
    train = FlotDataset.FlotDataset(conf, conf.dataTrainList, conf.transforms)
    dataset = torch.utils.data.DataLoader(train, batch_size = 32, num_workers = 1,
                                          shuffle = True, pin_memory = False)
    writer = SummaryWriter()
    probability = None
    labelsStack = None
    softmax_prob = torch.nn.Softmax()
    for data in dataset:
        writer.add_image('Image', data['img'], 0)
        writer.add_text('Text', 'text logged at step:'+str(1), 1)
        # plotSample(data)
        print("hi genie")
        out = conf.hyperparam.model(Variable(data['img']))
        labels = data['labels']
        if not probability: 
            probability = softmax_prob(out).data.cpu()
            labelsStack = data['labels']
        else:
            torch.stack((probability,softmax_prob(out).data.cpu()))
            torch.stack(labelsStack,data['labels'])


        # for i in range (labels.size()[0]):
        #     if (probability[0][i][0] < probability [0][i][0]) == labels
        # max_index, val = torch.max(probability,1)
        print ("out")
        print(out)
        #print(torch.max(probability,1))
        print ("probability")
        print(probability)
        break


    writer.close()