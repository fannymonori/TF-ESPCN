import tensorflow as tf
import os


class ESPCN:

    def __init__(self, input, scale, learning_rate):
        self.LR_input = input
        self.scale = scale
        self.learning_rate = learning_rate
        self.saver = ""

    def ESPCN_model(self):
        """
        Implementation of ESPCN: https://arxiv.org/abs/1609.05158

        Returns
        ----------
        Model
        """

        scale = self.scale
        channels = 1
        bias_initializer = tf.constant_initializer(value=0.1)
        initializer = tf.contrib.layers.xavier_initializer_conv2d()
        # initializer = tf.contrib.layers.variance_scaling_initializer()

        filters = [
            tf.Variable(initializer(shape=(5, 5, channels, 64)), name="f1"),  # (f1,n1) = (5,64)
            tf.Variable(initializer(shape=(3, 3, 64, 32)), name="f2"),  # (f2,n2) = (3,32)
            tf.Variable(initializer(shape=(3, 3, 32, channels * (scale * scale))), name="f3")  # (f3) = (3)
        ]

        bias = [
            tf.get_variable(shape=[64], initializer=bias_initializer, name="b1"),
            tf.get_variable(shape=[32], initializer=bias_initializer, name="b2"),
            tf.get_variable(shape=[channels * (scale * scale)], initializer=bias_initializer, name="b3")  # HxWxr^2
        ]

        l1 = tf.nn.conv2d(self.LR_input, filters[0], [1, 1, 1, 1], padding='SAME', name="conv1")
        l1 = l1 + bias[0]
        l1 = tf.nn.relu(l1)

        l2 = tf.nn.conv2d(l1, filters[1], [1, 1, 1, 1], padding='SAME', name="conv2")
        l2 = l2 + bias[1]
        l2 = tf.nn.relu(l2)

        l3 = tf.nn.conv2d(l2, filters[2], [1, 1, 1, 1], padding='SAME', name="conv3")
        l3 = l3 + bias[2]

        # Depth_to_space is equivalent to the pixel shuffle layer.
        out = tf.nn.depth_to_space(l3, scale, data_format='NHWC')

        out = tf.nn.tanh(out, name="NHWC_output")

        out_nchw = tf.transpose(out, [0, 3, 1, 2], name="NCHW_output")
        # out = tf.nn.relu(out, name="NHWC_output")

        self.saver = tf.train.Saver()

        return out

    def ESPCN_trainable_model(self, HR_out, HR_orig):
        psnr = tf.image.psnr(HR_out, HR_orig, max_val=1.0)

        loss = tf.losses.mean_squared_error(HR_orig, HR_out)

        # train_op = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)
        # train_op = tf.train.AdamOptimizer(learning_rate=0.00001).minimize(loss)
        train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)

        return loss, train_op, psnr
