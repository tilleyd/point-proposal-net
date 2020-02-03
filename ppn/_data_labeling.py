# ppn._data_labeling
# author: Duncan Tilley

def get_anchors(config):
    """
    Generates the anchor coordinates for a given image dimension.

    config
        The configuration dictionary. See ppn.config.ppn_config.
    """
    import numpy as np
    # create the anchor positions
    img_size = config['image_size']
    feature_size = config['feature_size']

    step = img_size / feature_size
    half_step = step * 0.5
    x = np.arange(half_step, img_size, step, dtype=np.float32)
    y = x.copy()
    return np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))])

def get_anchor_labels(anchors, coords, config):
    """
    Generates the anchor labels for tranining the PPN.

    Returns y_conf, y_reg.

    anchors
        The list of anchor coordinates generated from get_anchors().
    coords
        The list of ground truth point coordinates.
    config
        The configuration dictionary. See ppn.config.ppn_config.
    """
    import numpy as np
    r_near = config['r_near']
    r_far = config['r_far']
    img_size = config['image_size']
    feature_size = config['feature_size']

    step = img_size / feature_size
    halfstep = step * 0.5

    y_conf = np.full(anchors.shape[0], -1, dtype=np.int8)
    y_reg = np.zeros(anchors.shape)

    # For each point, find the nearest anchor and calculate the distance.
    # This ensures that most points have an associated anchor.
    for (x, y) in coords:
        x_norm = (x - halfstep) / step
        y_norm = (y - halfstep) / step
        r = int(np.round(y_norm))
        c = int(np.round(x_norm))
        anchor_index = r * feature_size + c
        y_conf[anchor_index] = 1
        y_reg[anchor_index][0] = (x - anchors[anchor_index][0]) / step
        y_reg[anchor_index][1] = (y - anchors[anchor_index][1]) / step

    # for each anchor, calculate the distances to each point
    count = 0
    for i in range(0, len(anchors)):
        x, y = anchors[i]
        x /= step
        y /= step
        distances = []
        for (px, py) in coords:
            px /= step
            py /= step
            distances.append(np.sqrt((x-px)**2 + (y-py)**2))
        if len(distances) > 0:
            near = np.argmin(distances)
            dist = distances[near]
            if dist <= r_near:
                y_conf[i] = 1
                px, py = coords[near]
                px /= step
                py /= step
                y_reg[i][0] = (px - x)
                y_reg[i][1] = (py - y)
            elif dist > r_far:
                y_conf[i] = 0

    # reshape for use in PPN training
    y_conf = np.reshape(y_conf, (feature_size, feature_size))
    y_reg = np.reshape(y_reg, (feature_size, feature_size) + (2,))
    return y_conf, y_reg

def get_fake_prediction(anchors, coords, config):
    """
    Generates the anchor labels with random noise to fake a good prediction.
    Used for testing non-model related code.

    anchors
        The list of anchor coordinates generated from get_anchors().
    coords
        The list of ground truth point coordinates.
    config
        The configuration dictionary. See ppn.config.ppn_config.
    """

    import numpy as np

    r_near = config['r_near']
    r_far = config['r_far']
    img_size = config['image_size']
    feature_size = config['feature_size']

    def noise():
        return np.random.normal(loc=0.0, scale=0.1)

    step = img_size / feature_size
    halfstep = step * 0.5

    y_conf = np.full(anchors.shape[0], 0.0, dtype=np.int8)
    y_reg = np.zeros(anchors.shape)

    # For each point, find the nearest anchor and calculate the distance.
    # This ensures that most points have an associated anchor.
    for (x, y) in coords:
        x_norm = (x - halfstep) / step
        y_norm = (y - halfstep) / step
        r = int(np.round(y_norm))
        c = int(np.round(x_norm))
        anchor_index = r * feature_size + c
        y_conf[anchor_index] = 1
        y_reg[anchor_index][0] = nosie() + (x - anchors[anchor_index][0]) / step
        y_reg[anchor_index][1] = noise() + (y - anchors[anchor_index][1]) / step

    # for each anchor, calculate the distances to each point
    for i in range(0, len(anchors)):
        x, y = anchors[i]
        x /= step
        y /= step
        distances = []
        for (px, py) in coords:
            px /= step
            py /= step
            distances.append(np.sqrt((x-px)**2 + (y-py)**2))
        near = np.argmin(distances)
        dist = distances[near]
        if dist <= r_near:
            y_conf[i] = 1
            px, py = coords[near]
            px /= step
            py /= step
            y_reg[i][0] = noise() + (px - x)
            y_reg[i][1] = noise() + (py - y)
        elif dist > r_far:
            y_conf[i] = 0

    # reshape for use in PPN training
    y_conf = np.reshape(y_conf, (feature_size, feature_size))
    y_reg = np.reshape(y_reg, (feature_size, feature_size) + (2,))
    return y_conf, y_reg
