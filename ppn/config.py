# ppn.config
# author: Duncan Tilley

def ppn_config():
    """
    Returns the default PPN configuration. Change the settings as needed.
    """
    import numpy as np
    return {
        # model parameters
        "image_size": 224,
        "feature_size": 7,

        # evaluation parameters
        "dist_thr": 0.5,
        "score_thr": 0.8,

        # training parameters
        "training": True,
        "epochs": 5,
        "batch_size": 16,
        "N_conf": 16.0,
        "N_reg": 16.0,
        "loss_function": 'focal', # or 'crossentropy'
        "focal_gamma": 2.0,
        "focal_pos_weight":  0.25,
        "r_far": np.sqrt(0.5*0.5 + 0.5*0.5),
        "r_near": np.sqrt(0.5*0.5 + 0.5*0.5),
        "r_nms": 0.45,
        "checkpoint_directory": 'checkpoints/',
        "drop_rate": 0.2
    }

def resnet_config():
    """
    Returns the default ResNet base model configuration. Change the settings as
    needed.
    """
    return {
        "initial_filters": 32,
        "initial_kernel": 7,
        "structure": [16, 8, 4, 2]
    }
