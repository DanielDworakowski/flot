import FlotDataset
import DefaultNNConfig
import torch
from DataUtil import plotSample
from tensorboardX import SummaryWriter

if __name__ == '__main__':
    #
    # The default configuration.
    conf = DefaultNNConfig.Config()
    train = FlotDataset.FlotDataset(conf, conf.dataTrainList, conf.transforms)
    dataset = torch.utils.data.DataLoader(train, batch_size = 32, num_workers = 1,
                                          shuffle = True, pin_memory = False)
    writer = SummaryWriter()
    for data in dataset:
        writer.add_image('Image', data['img'].cuda(), 0)
        writer.add_text('Text', 'text logged at step:'+str(1), 1)
        plotSample(data)
        break
    writer.close()
