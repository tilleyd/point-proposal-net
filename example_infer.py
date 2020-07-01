# example_infer
# author: Duncan Tilley

# This script shows how to use the PPN module for inferencing.

import tensorflow.compat.v1 as tf
tf.logging.set_verbosity(tf.logging.ERROR)

import ppn.config
import ppn.model
import ppn.data

# create the base and model configs
config = ppn.config.ppn_config()
resnet_config = ppn.config.resnet_config()

config['training'] = False

# read the images
data = ppn.data.Data(patch_images=False, image_directory='data-test/image', label_directory='data-test/label')

# create the PPN model from the saved weights
print('\nCreating model...\n')
model = ppn.model.PpnModel(ppn.model.get_resnet_constructor(resnet_config), config)
model.load_weights(config)

from ppn.metrics import precision_recall
import numpy as np

precision = np.zeros(data.num_images())
recall = np.zeros(data.num_images())
jrecall = {}
_, _, flux = data.get_image(0)
for j in flux:
    jrecall[j] = np.zeros(data.num_images())

print('[', ' '*50, '] %.2f%%'%(0.0), sep='', end='\r')
for i in range(0, data.num_images()):
    # infer the image
    patches, offsets = data.patch_image(i)
    predictions = model.infer(patches, offsets, config)

    _, coordinates, flux = data.get_image(i)
    precision[i], recall[i], jrec = precision_recall(predictions, coordinates, flux, config)
    for k, rec in jrec.items():
        jrecall[k][i] = rec

    perc = (i + 1) / data.num_images()
    bars = int(np.round(perc * 50))
    print('[', '='*bars, ' '*(50-bars), '] %.2f%%'%(perc*100.0), sep='', end='\r')
print('')

print('Precision: %.4f Recall: %.4f' %(np.mean(precision), np.mean(recall)))
for k, v in jrec.items():
    print('Hits at %.4f: %d' %(k, np.mean(v)))

data.display_image(data.num_images()-1, show_sources=True, predictions = predictions)
