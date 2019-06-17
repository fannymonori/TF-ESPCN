import tensorflow as tf
import os

class ESPCNmodel:
    def __init__(self, scale, LR_placeholder):
        #self.LR_input = tf.placeholder(tf.float32, [None, h, w, 1], name='images')
        self.LR_input = LR_placeholder
        self.scale = scale
        self.saver = ""

    def pixel_shift(self, L, scale):
        """
        Pixel shift layer. Implementation based on https://github.com/tetrachrome/subpixel
        Replaced squeeze with reshape for compatibility with other frameworks.
        Works with every batch size.
        It is equivalent to # out = tf.nn.depth_to_space(L, scale)

        Parameters
        ----------
        L: tensor
            input tensor shaped [batch_size x 17 x 17 x C*scale*scale]
        scale: int
            super-resolution scale

        Returns
        ----------
        Tensor with shape [batch_size x scale*17 x scale*17 x C]
        """

        # bsize, a, b, c = L.get_shape().as_list()
        # params = [-1, 1, 2, 1, 2, -1]
        #
        # PS_rs = tf.reshape(L, shape=(params[0], a, b, scale, scale), name="PS_reshape_1")
        # X1 = tf.split(PS_rs, a, params[1], name="PS_split_1")  # a, [bsize, b, r, r]
        # X2 = [tf.reshape(x, shape=(params[0], b, scale, scale), name="PS_reshape_2") for x in X1]
        # X3 = tf.concat(X2, params[2], name="PS_concat_1")  # bsize, b, a*r, r
        # X4 = tf.split(X3, b, params[3], name="PS_split_2")  # b, [bsize, a*r, r]
        # ####X5 = [tf.reshape(x, shape=(params[0], b * scale, scale), name="PS_reshape_3") for x in X4]
        # X5 = [tf.reshape(x, shape=(params[0], a * scale, scale), name="PS_reshape_3") for x in X4]
        # X6 = tf.concat(X5, params[4], name="PS_concat_2")  # bsize, a*r, b*r
        # out = tf.reshape(X6, (params[5], a * scale, b * scale, 1), name="PS_reshape_last")

        out = tf.nn.depth_to_space(L,scale,data_format='NHWC')
        out_nchw = tf.transpose(out, [0, 3, 1, 2], name="NCHW_output")

        return out


    def load_checkpoint(self, sess, ckpt_dir):
        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(sess, os.path.join(os.getcwd(), str(ckpt.model_checkpoint_path)))
            print("Checkpoint loaded!\n")
        else:
            print("Could not find checkpoint!\n")

        return

    def ESPCN_model(self):
        """
        Implementation of ESPCN: https://arxiv.org/abs/1609.05158

        Parameters
        ----------
        scale: int
            super-resolution scale
        LR_input:
            input LR dataset
        HR_output:
            output HR dataset

        Returns
        ----------
        Model
        """

        scale = self.scale
        channels = 1
        bias_initializer = tf.constant_initializer(value=0.1)
        initializer = tf.contrib.layers.xavier_initializer_conv2d()


        filters = [
            tf.Variable(initializer(shape=(5, 5, channels, 64)), name="f1"),  # (f1,n1) = (5,64)
            tf.Variable(initializer(shape=(3, 3, 64, 32)), name="f2"),  # (f2,n2) = (3,32)
            tf.Variable(initializer(shape=(3, 3, 32, channels * (scale * scale))), name="f3")  # (f3) = (3)
        ]

        bias = [
            tf.get_variable(shape=[64], initializer=bias_initializer, name="b1"),
            tf.get_variable(shape=[32], initializer=bias_initializer, name="b2"),
            tf.get_variable(shape=[channels * (scale * scale)], initializer=bias_initializer, name="b3") #HxWxr^2
        ]

        l1 = tf.nn.conv2d(self.LR_input, filters[0], [1, 1, 1, 1], padding='SAME', name="conv1")
        l1 = l1 + bias[0]
        l1 = tf.nn.relu(l1)

        l2 = tf.nn.conv2d(l1, filters[1], [1, 1, 1, 1], padding='SAME', name="conv2")
        l2 = l2 + bias[1]
        l2 = tf.nn.relu(l2)

        l3 = tf.nn.conv2d(l2, filters[2], [1, 1, 1, 1], padding='SAME', name="conv3")
        l3 = l3 + bias[2]

        out = self.pixel_shift(l3, scale)

        out = tf.nn.tanh(out, name="NHWC_output")

        self.saver = tf.train.Saver()

        return out

    def ESPCN_trainable_model(self, HR_input, HR_output):
        psnr = tf.image.psnr(HR_input, HR_output, max_val=1.0)

        loss = tf.reduce_mean(tf.square(HR_output - HR_input))

        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(0.01, global_step, 100000, 0.96, staircase=True)
        train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

        return loss, train_op, psnr