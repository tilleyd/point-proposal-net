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

# infer the first image
patches, offsets = data.patch_image(0)
predictions = model.infer(patches, offsets, config)

from ppn.metrics import precision_recall

_, coordinates, _ = data.get_image(0)
prec, rec = precision_recall(predictions, coordinates, config)
print('Precision: %.4f Recall: %.4f' %(prec, rec))

data.display_image(0, show_sources=True, predictions = predictions)
