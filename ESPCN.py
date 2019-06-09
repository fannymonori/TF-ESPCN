import tensorflow as tf

def pixel_shift(L, scale):
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

    bsize, a, b, c = L.get_shape().as_list()
    params = [-1, 1, 2, 1, 2, -1]

    PS_rs = tf.reshape(L, shape=(-1, a, b, scale, scale), name="PS_reshape_1")
    X1 = tf.split(PS_rs, a, params[1], name="PS_split_1")  # a, [bsize, b, r, r]
    X2 = [tf.reshape(x, shape=(-1, b, scale, scale), name="PS_reshape_2") for x in X1]
    X3 = tf.concat(X2, params[2], name="PS_concat_1")  # bsize, b, a*r, r
    X4 = tf.split(X3, b, params[3], name="PS_split_2")  # b, [bsize, a*r, r]
    X5 = [tf.reshape(x, shape=(-1, b * scale, scale), name="PS_reshape_3") for x in X4]
    X6 = tf.concat(X5, params[4], name="PS_concat_2")  # bsize, a*r, b*r
    out = tf.reshape(X6, (params[5], a * scale, b * scale, 1), name="PS_reshape_last")

    return out


def ESPCN_model(LR_input, HR_output, scale):
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

    channels = 1
    bias_initializer = tf.constant_initializer(value=0.1)

    filters = [
        tf.Variable(tf.random_normal([5, 5, channels, 64], stddev=0.1), name="f1"),  # (f1,n1) = (5,64)
        tf.Variable(tf.random_normal([3, 3, 64, 32], stddev=0.1), name="f2"),  # (f2,n2) = (3,32)
        tf.Variable(tf.random_normal([3, 3, 32, channels * (scale * scale)], stddev=0.1), name="f3")  # (f3) = (3)
    ]

    bias = [
        tf.get_variable(shape=[64], initializer=bias_initializer, name="b1"),
        tf.get_variable(shape=[32], initializer=bias_initializer, name="b2"),
        tf.get_variable(shape=[channels * (scale * scale)], initializer=bias_initializer, name="b3") #HxWxr^2
    ]

    l1 = tf.nn.conv2d(LR_input, filters[0], [1, 1, 1, 1], padding='SAME', name="conv1")
    l1 = l1 + bias[0]
    l1 = tf.nn.relu(l1)

    l2 = tf.nn.conv2d(l1, filters[1], [1, 1, 1, 1], padding='SAME', name="conv2")
    l2 = l2 + bias[1]
    l2 = tf.nn.relu(l2)

    l3 = tf.nn.conv2d(l2, filters[2], [1, 1, 1, 1], padding='SAME', name="conv3")
    l3 = l3 + bias[2]

    out = pixel_shift(l3, scale)

    out = tf.nn.tanh(out, name="NHWC_output")

    bsize, a, b, c = l3.get_shape().as_list()
    out_nchw = tf.reshape(out, (-1, 1, a * scale, b * scale), name="NCHW_output")

    psnr = tf.image.psnr(out, HR_output, max_val=1.0)

    loss = tf.losses.mean_squared_error(out, HR_output)
    train_op = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

    return loss, train_op, psnr
