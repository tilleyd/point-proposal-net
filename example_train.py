# example_train
# author: Duncan Tilley
#
# This script shows how to use the PPN module to train a model.

import ppn.config
import ppn.model
import ppn.data

# create the base and model configs
config = ppn.config.ppn_config()
resnet_config = ppn.config.resnet_config()

# create the datasets
data = ppn.data.Data(patch_images=True)
train, val, test = data.split_data([0.7, 0.15, 0.15], shuffle=True)

print('\nCreating data labels...\n')
train = ppn.data.create_labeled_set(train, config)
val = ppn.data.create_labeled_set(val, config)
test = ppn.data.create_labeled_set(test, config)

# create the PPN model
print('\nCreating model...\n')
model = ppn.model.PpnModel(ppn.model.get_resnet_constructor(resnet_config), config)

# train and evaluate the model on the dataset
print('\nStarting training...\n')
model.train(train, val, config)
# model.test(test, config)
