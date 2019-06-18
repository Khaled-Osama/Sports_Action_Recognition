'''
this file I3d contains the model architecture which uses Unit3D as a module of the architecture.
the output from the fuction build is the output of the final layer which has the shape(num_of_examples, num_of_classes)
'''
import tensorflow as tf
import sonnet as snt
from Unit3d import Unit3D

class InceptionI3D(snt.AbstractModule):

    def __init__(self,
                 spatial_squeeze=True,
                 num_of_classes=10):
        net_name = 'inception_i3d'
        super(InceptionI3D, self).__init__(name=net_name)
        self._spatial_squeeze = spatial_squeeze
        self._num_of_classes = num_of_classes

    def _build(self, inputs, is_training, dropout_keep_prob=1.0):
        VALID_ENDPOINTS = (
            'Conv3d_1a_7x7',
            'MaxPool3d_2a_3x3',
            'Conv3d_2b_1x1',
            'Conv3d_2c_3x3',
            'MaxPool3d_3a_3x3',
            'Mixed_3b',
            'Mixed_3c',
            'MaxPool3d_4a_3x3',
            'Mixed_4b',
            'Mixed_4c',
            'Mixed_4d',
            'Mixed_4e',
            'Mixed_4f',
            'MaxPool3d_5a_2x2',
            'Mixed_5b',
            'Mixed_5c',
            'Logits'
        )

        i = 0
        net = inputs
        net = Unit3D(output_channels=64,
                     kernel_shape=[7, 7, 7],
                     stride=[2, 2, 2],
                     name=VALID_ENDPOINTS[i])(net, is_training=is_training)
        i = i + 1

        net = tf.nn.max_pool3d(net, ksize=[1, 1, 3, 3, 1],
                               strides=[1, 1, 2, 2, 1],
                               padding=snt.SAME,
                               name=VALID_ENDPOINTS[i])

        i = i + 1

        net = Unit3D(output_channels=64,
                     kernel_shape=[1, 1, 1],
                     name=VALID_ENDPOINTS[i])(net, is_training=is_training)
        i = i + 1

        net = Unit3D(output_channels=192, kernel_shape=[3, 3, 3],
                     name=VALID_ENDPOINTS[i])(net, is_training=is_training)

        i = i + 1

        net = tf.nn.max_pool3d(net, ksize=[1, 1, 3, 3, 1], strides=[1, 1, 2, 2, 1],
                               padding=snt.SAME,
                               name=VALID_ENDPOINTS[i])

        i = i + 1

        with tf.variable_scope(VALID_ENDPOINTS[i]):
            with tf.variable_scope('Branch_0'):
                branch_0 = Unit3D(output_channels=64,
                                  kernel_shape=[1, 1, 1],
                                  name='Conv3d_0a_1x1')(net, is_training=is_training)

            with tf.variable_scope('Branch_1'):
                branch_1 = Unit3D(output_channels=96,
                                  kernel_shape=[1, 1, 1],
                                  name='Conv3d_0a_1x1')(net, is_training=is_training)
                branch_1 = Unit3D(output_channels=128,
                                  kernel_shape=[3, 3, 3],
                                  name='Conv3d_0b_3x3')(branch_1, is_training=is_training)

            with tf.variable_scope('Branch_2'):
                branch_2 = Unit3D(output_channels=16,
                                  kernel_shape=[1, 1, 1],
                                  name='Conv3d_0a_1x1')(net, is_training=is_training)

                branch_2 = Unit3D(output_channels=32,
                                  kernel_shape=[3, 3, 3],
                                  name='Conv3d_0b_3x3')(branch_2, is_training=is_training)

            with tf.variable_scope('Branch_3'):
                branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
                                            strides=[1, 1, 1, 1, 1], padding=snt.SAME,
                                            name='MaxPool3d_0a_3x3')
                branch_3 = Unit3D(output_channels=32, kernel_shape=[1, 1, 1],
                                  name='Conv3d_0b_1x1')(branch_3, is_training=is_training)
        net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)

        i = i + 1

        with tf.variable_scope(VALID_ENDPOINTS[i]):
            with tf.variable_scope('Branch_0'):
                branch_0 = Unit3D(output_channels=128,
                                  kernel_shape=[1, 1, 1],
                                  name='Conv3d_0a_1x1')(net, is_training=is_training)

            with tf.variable_scope('Branch_1'):
                branch_1 = Unit3D(output_channels=128,
                                  kernel_shape=[1, 1, 1],
                                  name='Conv3d_0a_1x1')(net, is_training=is_training)

                branch_1 = Unit3D(output_channels=192,
                                  kernel_shape=[3, 3, 3],
                                  name='Conv3d_0b_3x3')(branch_1, is_training=is_training)

            with tf.variable_scope('Branch_2'):
                branch_2 = Unit3D(output_channels=32, kernel_shape=[1, 1, 1],
                                  name='Conv3d_0a_1x1')(net, is_training=is_training)

                branch_2 = Unit3D(output_channels=96,
                                  kernel_shape=[3, 3, 3],
                                  name='Conv3d_0b_3x3')(branch_2, is_training=is_training)

            with tf.variable_scope('Branch_3'):
                branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
                                            strides=[1, 1, 1, 1, 1],
                                            padding=snt.SAME,
                                            name='MaxPool3d_0a_3x3')
                branch_3 = Unit3D(output_channels=64,
                                  kernel_shape=[1, 1, 1],
                                  name='Conv3d_0b_1x1')(branch_3, is_training=is_training)
        net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)

        i = i + 1

        net = tf.nn.max_pool3d(net,
                               ksize=[1, 3, 3, 3, 1],
                               strides=[1, 2, 2, 2, 1],
                               padding=snt.SAME,
                               name=VALID_ENDPOINTS[i])

        i = i + 1

        with tf.variable_scope(VALID_ENDPOINTS[i]):
            with tf.variable_scope('Branch_0'):
                branch_0 = Unit3D(output_channels=192,
                                  kernel_shape=[1, 1, 1],
                                  name='Conv3d_0a_1x1')(net, is_training=is_training)

            with tf.variable_scope('Branch_1'):
                branch_1 = Unit3D(output_channels=96,
                                  kernel_shape=[1, 1, 1],
                                  name='Conv3d_0a_1x1')(net, is_training=is_training)

                branch_1 = Unit3D(output_channels=208,
                                  kernel_shape=[3, 3, 3],
                                  name='Conv3d_0b_3x3')(branch_1, is_training=is_training)

            with tf.variable_scope('Branch_2'):
                branch_2 = Unit3D(output_channels=16, kernel_shape=[1, 1, 1],
                                  name='Conv3d_0a_1x1')(net, is_training=is_training)

                branch_2 = Unit3D(output_channels=48,
                                  kernel_shape=[3, 3, 3],
                                  name='Conv3d_0b_3x3')(branch_2, is_training=is_training)

            with tf.variable_scope('Branch_3'):
                branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
                                            strides=[1, 1, 1, 1, 1],
                                            padding=snt.SAME,
                                            name='MaxPool3d_0a_3x3')
                branch_3 = Unit3D(output_channels=64,
                                  kernel_shape=[1, 1, 1],
                                  name='Conv3d_0b_1x1')(branch_3, is_training=is_training)
        net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)

        i = i + 1

        with tf.variable_scope(VALID_ENDPOINTS[i]):
            with tf.variable_scope('Branch_0'):
                branch_0 = Unit3D(output_channels=160,
                                  kernel_shape=[1, 1, 1],
                                  name='Conv3d_0a_1x1')(net, is_training=is_training)

            with tf.variable_scope('Branch_1'):
                branch_1 = Unit3D(output_channels=112,
                                  kernel_shape=[1, 1, 1],
                                  name='Conv3d_0a_1x1')(net, is_training=is_training)

                branch_1 = Unit3D(output_channels=224,
                                  kernel_shape=[3, 3, 3],
                                  name='Conv3d_0b_3x3')(branch_1, is_training=is_training)

            with tf.variable_scope('Branch_2'):
                branch_2 = Unit3D(output_channels=24, kernel_shape=[1, 1, 1],
                                  name='Conv3d_0a_1x1')(net, is_training=is_training)

                branch_2 = Unit3D(output_channels=64,
                                  kernel_shape=[3, 3, 3],
                                  name='Conv3d_0b_3x3')(branch_2, is_training=is_training)

            with tf.variable_scope('Branch_3'):
                branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
                                            strides=[1, 1, 1, 1, 1],
                                            padding=snt.SAME,
                                            name='MaxPool3d_0a_3x3')
                branch_3 = Unit3D(output_channels=64,
                                  kernel_shape=[1, 1, 1],
                                  name='Conv3d_0b_1x1')(branch_3, is_training=is_training)
        net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)

        i = i + 1

        with tf.variable_scope(VALID_ENDPOINTS[i]):
            with tf.variable_scope('Branch_0'):
                branch_0 = Unit3D(output_channels=128,
                                  kernel_shape=[1, 1, 1],
                                  name='Conv3d_0a_1x1')(net, is_training=is_training)

            with tf.variable_scope('Branch_1'):
                branch_1 = Unit3D(output_channels=128,
                                  kernel_shape=[1, 1, 1],
                                  name='Conv3d_0a_1x1')(net, is_training=is_training)

                branch_1 = Unit3D(output_channels=256,
                                  kernel_shape=[3, 3, 3],
                                  name='Conv3d_0b_3x3')(branch_1, is_training=is_training)

            with tf.variable_scope('Branch_2'):
                branch_2 = Unit3D(output_channels=24, kernel_shape=[1, 1, 1],
                                  name='Conv3d_0a_1x1')(net, is_training=is_training)

                branch_2 = Unit3D(output_channels=64,
                                  kernel_shape=[3, 3, 3],
                                  name='Conv3d_0b_3x3')(branch_2, is_training=is_training)

            with tf.variable_scope('Branch_3'):
                branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
                                            strides=[1, 1, 1, 1, 1],
                                            padding=snt.SAME,
                                            name='MaxPool3d_0a_3x3')
                branch_3 = Unit3D(output_channels=64,
                                  kernel_shape=[1, 1, 1],
                                  name='Conv3d_0b_1x1')(branch_3, is_training=is_training)
        net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)

        i = i + 1

        with tf.variable_scope(VALID_ENDPOINTS[i]):
            with tf.variable_scope('Branch_0'):
                branch_0 = Unit3D(output_channels=112,
                                  kernel_shape=[1, 1, 1],
                                  name='Conv3d_0a_1x1')(net, is_training=is_training)

            with tf.variable_scope('Branch_1'):
                branch_1 = Unit3D(output_channels=144,
                                  kernel_shape=[1, 1, 1],
                                  name='Conv3d_0a_1x1')(net, is_training=is_training)

                branch_1 = Unit3D(output_channels=288,
                                  kernel_shape=[3, 3, 3],
                                  name='Conv3d_0b_3x3')(branch_1, is_training=is_training)

            with tf.variable_scope('Branch_2'):
                branch_2 = Unit3D(output_channels=32, kernel_shape=[1, 1, 1],
                                  name='Conv3d_0a_1x1')(net, is_training=is_training)

                branch_2 = Unit3D(output_channels=64,
                                  kernel_shape=[3, 3, 3],
                                  name='Conv3d_0b_3x3')(branch_2, is_training=is_training)

            with tf.variable_scope('Branch_3'):
                branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
                                            strides=[1, 1, 1, 1, 1],
                                            padding=snt.SAME,
                                            name='MaxPool3d_0a_3x3')
                branch_3 = Unit3D(output_channels=64,
                                  kernel_shape=[1, 1, 1],
                                  name='Conv3d_0b_1x1')(branch_3, is_training=is_training)
        net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)

        i = i + 1

        with tf.variable_scope(VALID_ENDPOINTS[i]):
            with tf.variable_scope('Branch_0'):
                branch_0 = Unit3D(output_channels=256,
                                  kernel_shape=[1, 1, 1],
                                  name='Conv3d_0a_1x1')(net, is_training=is_training)

            with tf.variable_scope('Branch_1'):
                branch_1 = Unit3D(output_channels=160,
                                  kernel_shape=[1, 1, 1],
                                  name='Conv3d_0a_1x1')(net, is_training=is_training)

                branch_1 = Unit3D(output_channels=320,
                                  kernel_shape=[3, 3, 3],
                                  name='Conv3d_0b_3x3')(branch_1, is_training=is_training)

            with tf.variable_scope('Branch_2'):
                branch_2 = Unit3D(output_channels=32, kernel_shape=[1, 1, 1],
                                  name='Conv3d_0a_1x1')(net, is_training=is_training)

                branch_2 = Unit3D(output_channels=128,
                                  kernel_shape=[3, 3, 3],
                                  name='Conv3d_0b_3x3')(branch_2, is_training=is_training)

            with tf.variable_scope('Branch_3'):
                branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
                                            strides=[1, 1, 1, 1, 1],
                                            padding=snt.SAME,
                                            name='MaxPool3d_0a_3x3')
                branch_3 = Unit3D(output_channels=128,
                                  kernel_shape=[1, 1, 1],
                                  name='Conv3d_0b_1x1')(branch_3, is_training=is_training)
        net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)

        i = i + 1

        net = tf.nn.max_pool3d(net, ksize=[1, 2, 2, 2, 1],
                               strides=[1, 2, 2, 2, 1],
                               padding=snt.SAME,
                               name=VALID_ENDPOINTS[i])
        i = i + 1

        with tf.variable_scope(VALID_ENDPOINTS[i]):
            with tf.variable_scope('Branch_0'):
                branch_0 = Unit3D(output_channels=256,
                                  kernel_shape=[1, 1, 1],
                                  name='Conv3d_0a_1x1')(net, is_training=is_training)

            with tf.variable_scope('Branch_1'):
                branch_1 = Unit3D(output_channels=160,
                                  kernel_shape=[1, 1, 1],
                                  name='Conv3d_0a_1x1')(net, is_training=is_training)

                branch_1 = Unit3D(output_channels=320,
                                  kernel_shape=[3, 3, 3],
                                  name='Conv3d_0b_3x3')(branch_1, is_training=is_training)

            with tf.variable_scope('Branch_2'):
                branch_2 = Unit3D(output_channels=32, kernel_shape=[1, 1, 1],
                                  name='Conv3d_0a_1x1')(net, is_training=is_training)

                branch_2 = Unit3D(output_channels=128,
                                  kernel_shape=[3, 3, 3],
                                  name='Conv3d_0a_3x3')(branch_2, is_training=is_training)

            with tf.variable_scope('Branch_3'):
                branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
                                            strides=[1, 1, 1, 1, 1],
                                            padding=snt.SAME,
                                            name='MaxPool3d_0a_3x3')
                branch_3 = Unit3D(output_channels=128,
                                  kernel_shape=[1, 1, 1],
                                  name='Conv3d_0b_1x1')(branch_3, is_training=is_training)
        net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)

        i = i + 1

        with tf.variable_scope(VALID_ENDPOINTS[i]):
            with tf.variable_scope('Branch_0'):
                branch_0 = Unit3D(output_channels=384,
                                  kernel_shape=[1, 1, 1],
                                  name='Conv3d_0a_1x1')(net, is_training=is_training)

            with tf.variable_scope('Branch_1'):
                branch_1 = Unit3D(output_channels=192,
                                  kernel_shape=[1, 1, 1],
                                  name='Conv3d_0a_1x1')(net, is_training=is_training)

                branch_1 = Unit3D(output_channels=384,
                                  kernel_shape=[3, 3, 3],
                                  name='Conv3d_0b_3x3')(branch_1, is_training=is_training)

            with tf.variable_scope('Branch_2'):
                branch_2 = Unit3D(output_channels=48, kernel_shape=[1, 1, 1],
                                  name='Conv3d_0a_1x1')(net, is_training=is_training)

                branch_2 = Unit3D(output_channels=128,
                                  kernel_shape=[3, 3, 3],
                                  name='Conv3d_0b_3x3')(branch_2, is_training=is_training)

            with tf.variable_scope('Branch_3'):
                branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
                                            strides=[1, 1, 1, 1, 1],
                                            padding=snt.SAME,
                                            name='MaxPool3d_0a_3x3')
                branch_3 = Unit3D(output_channels=128,
                                  kernel_shape=[1, 1, 1],
                                  name='Conv3d_0b_1x1')(branch_3, is_training=is_training)
        net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)

        i = i + 1

        with tf.variable_scope(VALID_ENDPOINTS[i]):
            net = tf.nn.avg_pool3d(net, ksize=[1, 2, 7, 7, 1],
                                   strides=[1, 1, 1, 1, 1],
                                   padding=snt.VALID)

            net = tf.nn.dropout(net, rate=(1 - dropout_keep_prob))
            logits = Unit3D(output_channels=10,
                            kernel_shape=[1, 1, 1],
                            activation_fn=None,
                            use_batch_norm=False,
                            use_bias=True,
                            name='Conv3d_0c_1x1')(net, is_training=is_training)

            logits = tf.squeeze(logits, [2, 3], name='SpatialSqueeze')
            averaged_logits = tf.reduce_mean(logits, axis=1)

        return averaged_logits