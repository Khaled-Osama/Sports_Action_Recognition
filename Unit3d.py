'''
this file *Unit3d* contain a some operations concatenated together which acts as one operation.
these operations are Conv3D, batch_normalization, and activation function.
'''

import sonnet as snt
import tensorflow as tf

class Unit3D(snt.AbstractModule):

    def __init__(self, output_channels,
                 kernel_shape=(1, 1, 1),
                 stride=(1, 1, 1),
                 activation_fn=tf.nn.relu,
                 use_batch_norm=True,
                 use_bias=False,
                 name='unit_3d'):

        super(Unit3D, self).__init__(name=name)
        self._output_channels = output_channels
        self._kernel_shape = kernel_shape
        self._stride = stride
        self._use_batch_norm = use_batch_norm
        self._activation_fn = activation_fn
        self._use_bias = use_bias

    def _build(self, inputs, is_training):
        net = snt.Conv3D(output_channels=self._output_channels,
                         kernel_shape=self._kernel_shape,
                         stride=self._stride,
                         padding=snt.SAME,
                         use_bias=self._use_bias)(inputs)

        if self._use_batch_norm:
            bn = snt.BatchNorm()
            net = bn(net, is_training=is_training, test_local_stats=False)
        if self._activation_fn is not None:
            net = self._activation_fn(net)
        return net