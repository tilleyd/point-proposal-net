# ppn._base_resnet
# author: Duncan Tilley

#
# layer helper functions
#

def _conv2d(inputs, filters, kernel, stride, drop_rate):
    import tensorflow.compat.v1 as tf

    x = tf.layers.conv2d(
        inputs = inputs,
        filters=filters,
        kernel_size=kernel,
        strides=stride,
        padding='same',
        use_bias=False,
        kernel_initializer=tf.variance_scaling_initializer(),
        data_format='channels_first')
    return tf.nn.dropout(x, rate=drop_rate)


def _batch_norm(inputs, training):
    import tensorflow.compat.v1 as tf

    DECAY = 0.997
    EPSILON = 1e-5
    return tf.layers.batch_normalization(
        inputs=inputs,
        axis=1, # for channel-first, 3 for channel-last
        momentum=DECAY,
        epsilon=EPSILON,
        center=True,
        scale=True,
        training=training,
        fused=True)


def _pool_max(inputs, kernel, stride):
    import tensorflow.compat.v1 as tf

    return tf.layers.max_pooling2d(
        inputs=inputs,
        pool_size=kernel,
        strides=stride,
        padding='same',
        data_format='channels_first')


def _resnet_block(inputs, filters, training, drop_rate, first):
    """
    Defines a single resnet block of 2 conv layers.

    inputs
        The input value (used as identity shortcut).
    filters
        The number of filters in each layer.
    training
        Whether the model is being constructed for training or inference.
    drop_rate
        Dropout rate scalar.
    first
        If this is the first block in the layer. If true, the first conv will
        have a stride of 2 to recude the size, and will apply a projection
        shortcut instead of an identity shortcut.
    """
    import tensorflow.compat.v1 as tf

    shortcut = inputs
    stride = 1
    if first:
        # reduce size and use projection shortcut
        stride = 2
        shortcut = _conv2d(inputs=shortcut, filters=filters, kernel=1, stride=2, drop_rate=drop_rate)
        shortcut = _batch_norm(shortcut, training)

    inputs = _conv2d(inputs=inputs, filters=filters, kernel=3, stride=stride, drop_rate=drop_rate)
    inputs = _batch_norm(inputs, training)
    inputs = tf.nn.relu(inputs)

    inputs = _conv2d(inputs=inputs, filters=filters, kernel=3, stride=1, drop_rate=drop_rate)
    inputs = _batch_norm(inputs, training)
    inputs += shortcut
    inputs = tf.nn.relu(inputs)

    return inputs


def _resnet_layer(inputs, filters, layers, training, drop_rate):
    # the first block reduces the size and uses projection shortcut
    inputs = _resnet_block(inputs, filters, training, drop_rate, first=True)
    for _ in range(1, layers):
        inputs = _resnet_block(inputs, filters, training, drop_rate, first=False)
    return inputs


#
# class definition
#

def ResNetBaseModel(inputs, drop_rate, resnet_config):
    """
    Constructs the PPN ResNet base layers.

    NOTE: This is a factory function but does not match the prototype required
    by the PPN. Use get_resnet_constructor(resnet_config) to bind the
    resnet_config parameter.

    inputs
        The input tensor, presumably of shape (?, chan, image_h, image_w).
    drop_rate
        Dropout rate scalar.
    resnet_config
        The ResNet configuration dictionary. See ppn.config.resnet_config.
    """
    import tensorflow.compat.v1 as tf

    init_filters = resnet_config['initial_filters']
    # initial conv and max pool layer
    x = _conv2d(inputs=inputs, filters=init_filters, kernel=resnet_config['initial_kernel'], stride=2, drop_rate=drop_rate)
    x = _batch_norm(x, True)
    x = tf.nn.relu(x)
    # x = _pool_max(inputs=x, kernel=3, stride=2)

    # resnet blocks
    # 2 of each conv block, according to the 18-layer architecture
    structure = resnet_config['structure']
    for i, layers in enumerate(structure):
        filters = init_filters * (2**i)
        x = _resnet_layer(
            inputs=x,
            filters=filters,
            layers=layers,
            drop_rate=drop_rate,
            training=True)

    return x

    # stop here for the base model, the rest of the resnet implementation
    # is not needed

    # do a global average pooling (i.e. average each channel), but use
    # reduce mean for improved performance
    #x = tf.reduce_mean(input_tensor=x, axis=[2, 3], keepdims=True)
    #x = tf.squeeze(x, [2, 3])

    # final fc layer
    #x = tf.layers.dense(inputs=x, units=1)


def get_resnet_constructor(resnet_config):
    """
    Returns a constructor for the ResNet base layers with the given config.

    resnet_config
        The ResNet configuration dictionary. See ppn.config.resnet_config.
    """
    return lambda inputs, drop_rate: ResNetBaseModel(inputs, drop_rate, resnet_config)
