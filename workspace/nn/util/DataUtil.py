import matplotlib.pyplot as plt
import torch
from torchvision import transforms, utils

def plotSample(sample):
    ''' Plot an image batch.

    '''
    images_batch = sample['img']
    batch_size = len(images_batch)
    im_size = images_batch.size(2)

    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.show()

class Rescale(object):
    '''Rescale the image in a sample to a given size.
    http://pytorch.org/tutorials/beginner/data_loading_tutorial.html

    Args:
        output_size (tuple or tuple): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    '''

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, labels = sample['img'], sample['labels']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
        #
        # New height and width
        new_h, new_w = int(new_h), int(new_w)
        #
        # Resize the image.
        img = transform.resize(image, (new_h, new_w))
        #
        # Can add label transformation here.
        return {'img': img, 'labels': labels}


class RandomCrop(object):
    '''Crop randomly the image in a sample.
    http://pytorch.org/tutorials/beginner/data_loading_tutorial.html


    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    '''

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, labels = sample['img'], sample['labels']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        labels = labels - [left, top]

        return {'img': image, 'labels': labels}


class ToTensor(object):
    '''Convert ndarrays in sample to Tensors.
    http://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    '''

    def __call__(self, sample):
        image, labels = sample['img'], sample['labels']
        #
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image[:,:,0:3] # Strip the alpha channel.
        image = image.transpose((2, 0, 1))
        return {'img': torch.from_numpy(image),
                'labels': torch.from_numpy(labels.as_matrix())}


class Normalize(object):
    '''Normalizes an image.
    http://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    '''

    def __init__(self, means, variances):
        assert len(means) == 3
        assert len(variances) == 3
        self.norm = transforms.Normalize(means, variances)

    def __call__(self, sample):
        sample['img'] = self.norm(sample['img'])
        return sample
