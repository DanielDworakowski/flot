import numpy as np
from PIL import Image

# Given image path, load image as RGB and calculate plane mean
# Returns tuple containing mean value of RGB plane
def getRGBMean(img_path):
    img = Image.open(img_path).convert('RGB')
    r, g, b = img.split()

    r_mean = np.array(r).mean()
    g_mean = np.array(g).mean()
    b_mean = np.array(b).mean()

    return r_mean, g_mean, b_mean

# Given image path, load image as RGB and calculate plane standard deviation
# Returns tuple containing std dev value of RGB plane
def getRGBStdDev(img_path):
    img = Image.open(img_path).convert('RGB')
    r, g, b = img.split()

    r_stddev = np.array(r).std()
    g_stddev = np.array(g).std()
    b_stddev = np.array(b).std()

    return r_stddev, g_stddev, b_stddev

# Given image path, load image as RGB and calculate plane mean and standard deviation
# Returns tuple of (mean, std dev) tuple value of RGB plane
def getRGBStat(img_path):
    img = Image.open(img_path).convert('RGB')
    r, g, b = img.split()

    r_mean = np.array(r).mean()
    g_mean = np.array(g).mean()
    b_mean = np.array(b).mean()

    r_stddev = np.array(r).std()
    g_stddev = np.array(g).std()
    b_stddev = np.array(b).std()

    return (r_mean, r_stddev), (g_mean, g_stddev), (b_mean, b_stddev)
