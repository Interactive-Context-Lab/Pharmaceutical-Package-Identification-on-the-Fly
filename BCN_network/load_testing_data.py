from glob import glob
from utils import *
from six.moves import xrange
import numpy as np

def load_imgs(path, is_grayscale, batch_size):
    sample_files = glob(path)
    for i in range(len(sample_files)):
        sample_files[i] = sample_files[i].replace("\\", "/")

    n = [int(i) for i in map(lambda x: x.split('/')[-1].split('.jpg')[0], sample_files)]
    sample_files = [x for (y, x) in sorted(zip(n, sample_files))]

    sample = [load_data(sample_file, is_test=True) for sample_file in sample_files]

    if (is_grayscale):
        sample_images = np.array(sample).astype(np.float32)[:, :, :, None]
    else:
        sample_images = np.array(sample).astype(np.float32)

    sample_images = [sample_images[i:i + batch_size]
                     for i in xrange(0, len(sample_images), batch_size)]
    sample_images = np.array(sample_images)

    return sample_images