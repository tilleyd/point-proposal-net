# ppn._model_ppn
# author: Duncan Tilley

#
# helper functions
#

def _conv2d(inputs, filters, kernel, drop_rate, stride=1, bias_init=None):
    import tensorflow.compat.v1 as tf
    if bias_init is None:
        bias_init = tf.zeros_initializer()
    x = tf.layers.conv2d(
        inputs = inputs,
        filters=filters,
        kernel_size=kernel,
        strides=stride,
        padding='same',
        use_bias=True,
        bias_initializer=bias_init,
        kernel_initializer=tf.variance_scaling_initializer(),
        data_format='channels_first')
    return tf.nn.dropout(x, rate=drop_rate)

def _binary_focal_loss_with_logits(labels, logits, gamma, pos_weight=None):
    """
    Compute focal loss from logits.

    Adapted from github.com/artemmavrin/focal-loss for TF 1.14.
    """
    import tensorflow.compat.v1 as tf
    # Compute probabilities for the positive class
    p = tf.math.sigmoid(logits)

    # The labels and logits tensors' shapes need to be the same for the
    # built-in cross-entropy functions. Since we want to allow broadcasting,
    # we do some checks on the shapes and possibly broadcast explicitly
    # Note: tensor.shape returns a tf.TensorShape, whereas tf.shape(tensor)
    # returns an int tf.Tensor; this is why both are used below
    labels_shape = labels.shape
    logits_shape = logits.shape
    if not labels_shape.is_fully_defined() or labels_shape != logits_shape:
        labels_shape = tf.shape(labels)
        logits_shape = tf.shape(logits)
        shape = tf.broadcast_dynamic_shape(labels_shape, logits_shape)
        labels = tf.broadcast_to(labels, shape)
        logits = tf.broadcast_to(logits, shape)
    if pos_weight is None:
        loss_func = tf.nn.sigmoid_cross_entropy_with_logits
    else:
        from functools import partial
        loss_func = partial(tf.nn.weighted_cross_entropy_with_logits,
                            pos_weight=pos_weight)
    loss = loss_func(labels=labels, logits=logits)
    if abs(gamma) < 0.00001:
        # no modulation (the loss returns NaN with ** 0)
        return loss
    else:
        modulation_pos = (1 - p) ** gamma
        modulation_neg = p ** gamma
        mask = tf.dtypes.cast(labels, dtype=tf.bool)
        modulation = tf.where(mask, modulation_pos, modulation_neg)
        return modulation * loss

def _loss_function(conf_gt, conf_logits, reg_gt, reg_logits, config):
    """
    Creates the PPN loss function.

    Returns (conf_loss, point_loss)

    conf_gt:
        Ground truth confidence, i.e. 1 for close anchors, 0 for anchors
        that are too far off and -1 for anchors to be ignored. Must have
        shape (?, fh, fw, k).
    conf_logits:
        PPN confidence output, must have shape (?, fh, fw, k).
    reg_gt:
        Ground truth point offsets, need only have valid values for the
        anchors with conf_gt of 1. Must have shape (?, fh, fw, 2k).
    reg_logits:
        PPN anchor offset output, must have shape (?, fh, fw, 2k).
    config
            The configuration dictionary. See ppn.config.ppn_config.
    """
    import tensorflow.compat.v1 as tf

    # mask out the invalid anchors:
    #     only penalize confidence of valid (i.e. not ignored) anchors
    #     only penalize points of positive anchors
    valid_mask = tf.stop_gradient(tf.not_equal(conf_gt, -1))
    pos_mask = tf.stop_gradient(tf.equal(conf_gt, 1))
    num_valid = tf.stop_gradient(tf.count_nonzero(valid_mask, dtype=tf.int32))
    num_pos = tf.stop_gradient(tf.count_nonzero(pos_mask, dtype=tf.int32))
    valid_conf_gt = tf.boolean_mask(conf_gt, valid_mask)
    valid_conf_logits = tf.boolean_mask(conf_logits, valid_mask)
    pos_reg_gt = tf.boolean_mask(reg_gt, pos_mask)
    pos_reg_logits = tf.boolean_mask(reg_logits, pos_mask)

    if config['loss_function'] == 'crossentropy':
        # get the confidence loss using sigmoidal cross entropy
        conf_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.cast(valid_conf_gt, tf.float32),
            logits=valid_conf_logits)
    else:
        # get the confidence loss using focal loss
        conf_loss = _binary_focal_loss_with_logits(
            labels=tf.cast(valid_conf_gt, tf.float32),
            logits=valid_conf_logits,
            gamma=config['focal_gamma'],
            pos_weight=config['focal_pos_weight'])
        if config['focal_normalized']:
            # normalize according to number of positive anchors
            conf_loss = conf_loss / tf.cast(num_valid, tf.float32)
    conf_loss = tf.reduce_sum(conf_loss)

    # get the point loss using MSE
    point_loss = tf.losses.mean_squared_error(
        labels=pos_reg_gt,
        predictions=pos_reg_logits,
        reduction=tf.losses.Reduction.SUM)

    # zero out the losses if there were no valid points
    conf_loss = tf.where(tf.equal(num_valid, 0),
                            0.0,
                            conf_loss,
                            name='conf_loss')
    point_loss = tf.where(tf.equal(num_pos, 0),
                            0.0,
                            point_loss,
                            name='point_loss')

    # normalize losses to contribute equally and add
    N_conf, N_reg = config['N_conf'], config['N_reg']
    return ((1.0/N_conf) * conf_loss, (1.0/N_reg) * point_loss)

def _prec_rec(conf_gt, conf_out, reg_gt, reg_out, config):
    """
    Creates precision and recall metrics.

    Returns (precision, recall)

    conf_gt
        Ground truth confidence, i.e. 1 for close anchors, 0 for anchors
        that are too far off and -1 for anchors to be ignored. Must have
        shape (?, fh, fw, k).
    conf_out
        PPN confidence output, must have shape (?, fh, fw, k).
    reg_gt
        Ground truth point offsets, need only have valid values for the
        anchors with conf_gt of 1. Must have shape (?, fh, fw, 2k).
    reg_out
        PPN anchor offset output, must have shape (?, fh, fw, 2k).
    config
        The configuration dictionary. See ppn.config.ppn_config.
    """
    import tensorflow.compat.v1 as tf

    score_thr = config['score_thr']
    dist_thr = config['dist_thr']

    # mask positive outputs and ground truths
    gt_pos_mask = tf.equal(conf_gt, 1)
    gt_pos_count = tf.count_nonzero(gt_pos_mask, dtype=tf.int32)
    out_pos_mask = tf.greater_equal(conf_out, score_thr)
    out_pos_count = tf.count_nonzero(out_pos_mask, dtype=tf.int32)

    # calculate regression distances
    dist = reg_gt - reg_out
    dist *= dist
    dist = tf.reduce_sum(dist, axis=3)
    close_mask = tf.less_equal(dist, dist_thr*dist_thr) # uses squared distance

    # calculate number of correct predictions
    correct_mask = tf.logical_and(tf.logical_and(gt_pos_mask, out_pos_mask),
                                  close_mask)
    correct_count = tf.count_nonzero(correct_mask, dtype=tf.float32)

    precision = tf.where(tf.equal(out_pos_count, 0),
                    0.0,
                    correct_count / tf.cast(out_pos_count, tf.float32))
    recall = tf.where(tf.equal(out_pos_count, 0),
                    0.0,
                    correct_count / tf.cast(gt_pos_count, tf.float32))

    return tf.stop_gradient(precision), tf.stop_gradient(recall)

def _nms(points, scores, dist_thr, score_thr):
    """
    Creates the non-max suppression step for inferencing.

    Returns points and scores after suppression, with shapes (?, 2) and (?),
    respectively.

    points
        List of predicted points with shape (fh*fw, 2)
    scores
        List of predicted scores with shape (fh*fw)
    dist_thr
        The radius within which to suppress non-max predictions.
    score_thr
        The minimum score that a valid prediction must have.
    """
    import tensorflow.compat.v1 as tf

    # use reciprocal of squared distance as threshold
    if dist_thr == 0.0:
        dist_thr = float('inf')
    else:
        dist_thr = 1.0 / (dist_thr*dist_thr)

    # calculate distances
    r = tf.reduce_sum(points*points, 1)
    # turn r into column vector
    r = tf.reshape(r, [-1, 1])
    D = r - 2*tf.matmul(points, tf.transpose(points)) + tf.transpose(r)

    nms_indices = tf.image.non_max_suppression_overlaps(
        overlaps=1.0 / D, # distances, not really overlaps
        scores=scores,
        max_output_size=tf.shape(points)[0],
        overlap_threshold=dist_thr,
        score_threshold=score_thr)

    proposed_points = tf.gather(points, nms_indices)
    proposed_scores = tf.gather(scores, nms_indices)

    return tf.stop_gradient(proposed_points), tf.stop_gradient(proposed_scores)

#
# class definition
#

class PpnModel(object):
    """Point proposal network model class."""

    def __init__(self, base_constructor, config):
        """
        Constructs the model.

        base_constructor
            A callable that returns the feature layer of the base CNN. Must
            have prototype of fn(input, drop_rate).
        config
            The configuration dictionary. See ppn.config.ppn_config.
        """
        import numpy as np
        import tensorflow.compat.v1 as tf

        img_size = config["image_size"]
        ft_size = config["feature_size"]
        training = config["training"]

        # create data placeholders
        self.ph_batch_size = tf.placeholder(dtype=tf.int64)
        self.ph_train_size = tf.placeholder(dtype=tf.int64)
        self.ph_drop_rate = tf.placeholder_with_default(0.0, shape=())

        self.in_image = tf.placeholder(dtype=tf.float32,
                                       shape=(None, 1, img_size, img_size))
        self.in_offsets = tf.placeholder_with_default(np.zeros((1, 2),
                                                      dtype=np.float32),
                                                      shape=(None, 2))

        if training:
            self.in_conf = tf.placeholder_with_default(np.zeros((1, ft_size, ft_size),
                                                                dtype=np.float32),
                                                       shape=(None, ft_size, ft_size))
            self.in_reg = tf.placeholder_with_default(np.zeros((1, ft_size, ft_size, 2),
                                                               dtype=np.float32),
                                                      shape=(None, ft_size, ft_size, 2))

            # create the data pipeline with training sets
            train_dataset = tf.data.Dataset \
                    .from_tensor_slices((self.in_image, self.in_conf, self.in_reg)) \
                    .shuffle(self.ph_train_size) \
                    .batch(self.ph_batch_size, drop_remainder=True)
            feed_dataset = tf.data.Dataset \
                    .from_tensor_slices((self.in_image,
                                        self.in_conf,
                                        self.in_reg)) \
                    .batch(self.ph_batch_size, drop_remainder=True)
            iterator = tf.data.Iterator.from_structure(train_dataset.output_types,
                                                       train_dataset.output_shapes)
            self.it_image, self.it_conf, self.it_reg = iterator.get_next()

            self.train_iterator = iterator.make_initializer(train_dataset)
            self.feed_iterator = iterator.make_initializer(feed_dataset)
        else:
            # create the data pipeline without training sets
            feed_dataset = tf.data.Dataset \
                    .from_tensor_slices(self.in_image) \
                    .batch(self.ph_batch_size, drop_remainder=False)
            iterator = tf.data.Iterator.from_structure(feed_dataset.output_types,
                                                       feed_dataset.output_shapes)
            self.it_image = iterator.get_next()

            self.feed_iterator = iterator.make_initializer(feed_dataset)

        # explicitly initialize the base layers
        x = base_constructor(inputs=self.it_image, drop_rate=self.ph_drop_rate)

        # set up the PPN layers
        x = _conv2d(inputs=x, filters=512, kernel=3, drop_rate=self.ph_drop_rate)

        if config['loss_function'] != 'crossentropy' or config['focal_gamma'] < 0.00001:
            # no focal loss, use zero-initialized biases
            conf = _conv2d(inputs=x, filters=1, kernel=1, drop_rate=self.ph_drop_rate)
        else:
            # focal loss, initialize biases to -log((1-pi)/pi) with pi=0.01
            conf = _conv2d(inputs=x, filters=1, kernel=1, drop_rate=self.ph_drop_rate,
                           bias_init=tf.constant_initializer(-1.99563519))
        # reshape from (?, 1, fh, fw) to (?, fh, fw)
        conf_logits = tf.squeeze(tf.transpose(conf, [0, 2, 3, 1]))

        reg = _conv2d(inputs=x, filters=2, kernel=1, drop_rate=self.ph_drop_rate)
        # reshape from (?, 2, fh, fw) to (?, fh, fw, 2)
        reg = tf.transpose(reg, [0, 2, 3, 1])

        self.trainable = training
        self.out_conf = tf.nn.sigmoid(conf_logits)
        self.out_reg = reg
        self.initialized = False

        if training:
            # set up labels, loss and optimiser
            loss_conf, loss_reg = _loss_function(self.it_conf, conf_logits,
                                                 self.it_reg, self.out_reg,
                                                 config)
            optimiser = tf.train.AdamOptimizer()
            self.loss_conf = loss_conf
            self.loss_reg = loss_reg
            self.loss = self.loss_conf + self.loss_reg
            self.precision, self.recall = _prec_rec(self.it_conf, self.out_conf,
                                                    self.it_reg, self.out_reg,
                                                    config)
            self.trainer = optimiser.minimize(self.loss)

        # set up non-max suppression for prediction
        # NOTE: the nms assumes that all images in the batch are patches of
        #       a larger image, thus flattening all predictions

        num_img = tf.shape(self.out_conf)[0]
        reg_list = tf.reshape(self.out_reg, (num_img, ft_size*ft_size, 2))

        # revert from logical offsets to pixel offsets
        step = img_size / ft_size
        reg_list = reg_list*step

        # calculate the pixel coordinates of anchors
        from ppn._data_labeling import get_anchors
        anchors = get_anchors(config)

        # add anchors to revert to per-patch pixel coordinates
        anchor_add = tf.tile(tf.constant(np.array([anchors])),
                             [num_img, 1, 1])
        reg_list = reg_list + anchor_add

        # add patch offsets to revert to image pixel coordinates
        offset_add = tf.tile(tf.reshape(self.in_offsets, (num_img, 1, 2)),
                             [1, ft_size*ft_size, 1])
        reg_list = reg_list + offset_add


        # completely flatten everything
        conf_list = tf.reshape(self.out_conf, (num_img*ft_size*ft_size,))
        reg_list = tf.reshape(reg_list, (num_img*ft_size*ft_size, 2))

        # apply NMS and store results before and after
        self.unsup_points, self.unsup_conf = reg_list, conf_list
        self.sup_points, self.sup_conf = _nms(reg_list,
                                              conf_list,
                                              config['r_nms'] * step,
                                              config['score_thr'])


    def train(self, train_set, val_set, config):
        """
        Starts the training loop of the model.

        train_set
            The training set dictionary. Must have ['x'], ['y_conf'] and
            ['y_reg'] keys defined.
        val_set
            Similar to the training set dictionary. Used for validation after
            each epoch.
        config
            The configuration dictionary. See ppn.config.ppn_config.
        """
        import tensorflow.compat.v1 as tf
        import os


        if not self.trainable:
            print('Error: model and base model must be set up for training')
            return

        if not self.initialized:
            print('\nInitializing model...\n')
            # create the tensorflow session
            self.session = tf.Session()
            # initialize variables
            self.session.run(tf.global_variables_initializer())
            self.initialized = True

        # set up saver to save best validation set
        saver = tf.train.Saver()
        save_dir = config['checkpoint_directory']
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, 'best_validation')

        best_validation_loss = -1
        batch_size = config['batch_size']
        train_size = len(train_set['x'])

        print('\nStarting training...\n')

        # perform epochs
        for t in range(1, config['epochs'] + 1):

            # feed the training set to the train iterator
            self.session.run(self.train_iterator, {
                self.ph_train_size: train_size,
                self.ph_batch_size: batch_size,
                self.in_image: train_set['x'],
                self.in_conf: train_set['y_conf'],
                self.in_reg: train_set['y_reg']
            })

            # train an epoch
            conf_loss_train, reg_loss_train = 0.0, 0.0
            prec_train, rec_train = 0.0, 0.0
            count = 0
            try:
                while True:
                    _, cl, rl, prec, rec = self.session.run(
                        (self.trainer,
                         self.loss_conf, self.loss_reg,
                         self.precision, self.recall),
                        {self.ph_drop_rate: config['drop_rate']})
                    conf_loss_train += cl
                    reg_loss_train += rl
                    prec_train += prec * 100.0
                    rec_train += rec * 100.0
                    count += 1
            except tf.errors.OutOfRangeError:
                pass
            if count > 1:
                conf_loss_train /= count
                reg_loss_train /= count
                prec_train /= count
                rec_train /= count

            # feed the validation set to the feed iterator
            self.session.run(self.feed_iterator, {
                self.ph_batch_size: batch_size,
                self.in_image: val_set['x'],
                self.in_conf: val_set['y_conf'],
                self.in_reg: val_set['y_reg']
            })
            # validate on entire validation set
            conf_loss_val, reg_loss_val = 0.0, 0.0
            prec_val, rec_val = 0.0, 0.0
            count = 0
            try:
                while True:
                    cl, rl, prec, rec = self.session.run(
                        (self.loss_conf, self.loss_reg,
                         self.precision, self.recall))
                    conf_loss_val += cl
                    reg_loss_val += rl
                    prec_val += prec * 100.0
                    rec_val += rec * 100.0
                    count += 1
            except tf.errors.OutOfRangeError:
                pass
            if count > 1:
                conf_loss_val /= count
                reg_loss_val /= count
                prec_val /= count
                rec_val /= count

            print('epoch %d' %(t), end='')
            # check for improvement
            if best_validation_loss == -1 or (conf_loss_val + reg_loss_val) < best_validation_loss:
                best_validation_loss = conf_loss_val + reg_loss_val
                saver.save(sess=self.session, save_path=save_path)
                print('\t*', end='')
            print('')

            print('trn:\tcl=%.4f\trl=%.4f\tprc=%.4f\trcl=%.4f'
                  %(conf_loss_train, reg_loss_train, prec_train, rec_train))
            print('val:\tcl=%.4f\trl=%.4f\tprc=%.4f\trcl=%.4f'
                  %(conf_loss_val, reg_loss_val, prec_val, rec_val))


    def test(self, test_set, config):
        """
        Evaluates the test set.

        test_set
            The test set dictionary. Must have ['x'], ['y_conf'] and ['y_reg']
            keys defined.
        config
            The configuration dictionary. See ppn.config.ppn_config.
        """
        import tensorflow.compat.v1 as tf

        if not self.trainable:
            print('Error: model and base model must be set up for training')
            return

        if not self.initialized:
            print('Error: can\'t evaluate an uninitialized model (did you train or load weights?)')

        # feed the test set to the feed iterator
        self.session.run(self.feed_iterator, {
            self.ph_batch_size: config['batch_size'],
            self.in_image: test_set['x'],
            self.in_conf: test_set['y_conf'],
            self.in_reg: test_set['y_reg']
        })
        # test on entire test set
        conf_loss, reg_loss = 0.0, 0.0
        prec, rec = 0.0, 0.0
        count = 0
        try:
            while True:
                cl, rl, pr, rc = self.session.run(
                    (self.loss_conf, self.loss_reg,
                     self.precision, self.recall))
                conf_loss += cl
                reg_loss += rl
                prec += pr * 100.0
                rec += rc * 100.0
                count += 1
        except tf.errors.OutOfRangeError:
            pass
        if count > 1:
            conf_loss /= count
            reg_loss /= count
            prec /= count
            rec /= count

        print('test')
        print('\tcl=%.4f\trl=%.4f\tprc=%.4f\trcl=%.4f'
               %(conf_loss, reg_loss, prec, rec))


    def infer(self, images, offsets, config):
        """
        Evaluates a set of image patches.

        images
            The list of input image patches, (?, img_size, img_size).
        offsets
            The (x, y) offsets of the image corners, (?, 2).
        config
            The configuration dictionary. See ppn.config.ppn_config.
        """
        import numpy as np

        max_batch = config['batch_size']
        if images.shape[0] > max_batch:
            # split into multiple groups
            images = np.array_split(images, np.ceil(images.shape[0]/max_batch))
            offsets = np.array_split(offsets, np.ceil(offsets.shape[0]/max_batch))
            all_points = np.zeros((0, 2), dtype=np.float32)
            for i in range(0, len(images)):
                # initialize iterator with images
                self.session.run(self.feed_iterator, {
                    self.in_image: images[i],
                    self.ph_batch_size: images[i].shape[0]
                })
                # return results
                pred_points = self.session.run(self.sup_points, {
                    self.in_offsets: offsets[i]
                })
                all_points = np.concatenate([all_points, pred_points])
            return all_points
        else:
            # initialize iterator with images
            self.session.run(self.feed_iterator, {
                self.in_image: images,
                self.ph_batch_size: len(images)
            })
            # return results
            pred_points = self.session.run(self.sup_points, {self.in_offsets: offsets})
            return pred_points


    # def infer_no_nms(self, session, images, offsets, max_batch=128):
    #     """
    #     Loads the saved checkpoint and evaluates the image patches without NMS.

    #     session
    #         A tf.Session.
    #     images
    #         The list of input image patches, (?, img_size, img_size).
    #     offsets
    #         The (x, y) offsets of the image corners, (?, 2).
    #     max_batch
    #         The maximum number of patches that should be fed to the PPN
    #         at a time.
    #     """
    #     if images.shape[0] > max_batch:
    #         # split into multiple groups
    #         images = np.array_split(images, np.ceil(images.shape[0]/max_batch))
    #         offsets = np.array_split(offsets, np.ceil(offsets.shape[0]/max_batch))
    #         all_points = np.zeros((0, 2), dtype=np.float32)
    #         all_confs = np.zeros((0), dtype=np.float32)
    #         for i in range(0, len(images)):
    #             # initialize iterator with images
    #             session.run(self.feed_iterator, {
    #                 self.feed_image: images[i],
    #                 self.ph_batch_size: images[i].shape[0]
    #             })
    #             # return results
    #             pred_points, pred_confs = session.run(
    #                 (self.unsup_points, self.unsup_conf), {
    #                 self.in_offsets: offsets[i]
    #             })
    #             all_points = np.concatenate([all_points, pred_points])
    #             all_confs = np.concatenate([all_confs, pred_confs])
    #         return all_points
    #     else:
    #         # initialize iterator with images
    #         session.run(self.feed_iterator, {
    #             self.feed_image: images,
    #             self.ph_batch_size: len(images)
    #         })
    #         # return results
    #         pred_points, pred_confs = session.run(
    #                 (self.unsup_points, self.unsup_conf),
    #                 {self.in_offsets: offsets}
    #         )
    #         return pred_points, pred_confs


    # def benchmark(self, session, images, offsets, max_batch=128):
    #     """
    #     Loads the saved checkpoint and evaluates the image patches, returning
    #     the times taken to evaluate different parts.

    #     session
    #         A tf.Session.
    #     images
    #         The list of input image patches, (?, img_size, img_size).
    #     offsets
    #         The (x, y) offsets of the image corners, (?, 2).
    #     """
    #     import time

    #     if images.shape[0] > max_batch:
    #         # split into multiple groups
    #         images = np.array_split(images, np.ceil(images.shape[0]/max_batch))
    #         offsets = np.array_split(offsets, np.ceil(offsets.shape[0]/max_batch))

    #         # perform without nms
    #         before = time.time()
    #         for i in range(0, len(images)):
    #             session.run(self.feed_iterator, {
    #                 self.feed_image: images[i],
    #                 self.ph_batch_size: images[i].shape[0]
    #             })
    #             session.run((self.out_reg, self.out_conf))
    #         after = time.time()
    #         without_nms_time = after - before

    #         # perform with nms
    #         before = time.time()
    #         all_points = np.zeros((0, 2), dtype=np.float32)
    #         for i in range(0, len(images)):
    #             session.run(self.feed_iterator, {
    #                 self.feed_image: images[i],
    #                 self.ph_batch_size: images[i].shape[0]
    #             })
    #             pred_points = session.run(self.sup_points, {
    #                 self.in_offsets: offsets[i]
    #             })
    #             all_points = np.concatenate([all_points, pred_points])
    #         after = time.time()
    #         with_nms_time = after - before

    #         return without_nms_time, with_nms_time

    #     else:
    #         # perform without nms
    #         before = time.time()
    #         session.run(self.feed_iterator, {
    #             self.feed_image: images,
    #             self.ph_batch_size: len(images)
    #         })
    #         session.run((self.out_reg, self.out_conf))
    #         after = time.time()
    #         without_nms_time = after - before

    #         # perform with nms
    #         before = time.time()
    #         session.run(self.feed_iterator, {
    #             self.feed_image: images,
    #             self.ph_batch_size: len(images)
    #         })
    #         session.run(self.sup_points, {self.in_offsets: offsets})
    #         after = time.time()
    #         with_nms_time = after - before

    #         return without_nms_time, with_nms_time


    def load_weights(self, config):
        """
        Loads the weights from the checkpoint directory.

        config
            The configuration dictionary. See ppn.config.ppn_config.
        """
        import tensorflow.compat.v1 as tf
        import os


        if not self.initialized:
            # create the tensorflow session
            self.session = tf.Session()
            # initialize variables
            self.session.run(tf.global_variables_initializer())
            self.initialized = True

        # set up saver to load checkpoint
        saver = tf.train.Saver()
        save_dir = config['checkpoint_directory']
        if not os.path.exists(save_dir):
            raise Exception('Could not load weights from ' + save_dir)

        # reload weights
        save_path = os.path.join(save_dir, 'best_validation')
        saver.restore(sess=self.session, save_path=save_path)
