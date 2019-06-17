import tensorflow as tf
import pathlib
import ESPCN
import cv2
import numpy as np
import os
from sklearn.feature_extraction import image

from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib
from tensorflow.tools.graph_transforms import TransformGraph

# def load_dataset(path, scale):
#     """
#     Load the jpeg dataset for training.
#
#     Parameters
#     ----------
#     path: list of strings
#         Path of images
#     scale: int
#         Super-resolution scale
#     channels: int
#         Number of channels used for training
#     """
#
#     scale_factor = tf.constant(scale)
#     channels = 1
#     image = tf.read_file(path)
#     image = tf.image.decode_jpeg(image, channels=3)
#
#     image = tf.cast(image, tf.float32)
#     R, G, B = tf.unstack(image, 3, axis=2)
#
#     y = R * 0.299 + G * 0.587 + B * 0.114
#     image_preprocessed = tf.reshape(y, (tf.shape(image)[0], tf.shape(image)[1], 1)) / 255
#
#     X = tf.dtypes.cast((tf.shape(image_preprocessed)[0] / scale_factor), dtype=tf.int32) * scale_factor
#     Y = tf.dtypes.cast((tf.shape(image_preprocessed)[1] / scale_factor), dtype=tf.int32) * scale_factor
#     high = tf.image.crop_to_bounding_box(image_preprocessed, 0, 0, X, Y)
#
#     imgshape = tf.shape(high)
#     size = [imgshape[0] / scale_factor, imgshape[1] / scale_factor]
#     low = tf.image.resize_images(high, size=size, method=tf.image.ResizeMethod.BILINEAR)
#
#     hshape = tf.shape(high)
#     lshape = tf.shape(low)
#
#     high_r = tf.reshape(high, [1, hshape[0], hshape[1], channels])
#     low_r = tf.reshape(low, [1, lshape[0], lshape[1], channels])
#
#     slice_l = tf.image.extract_image_patches(low_r, [1, 17, 17, 1], [1, 14, 14, 1], [1, 1, 1, 1], "VALID")
#     slice_h = tf.image.extract_image_patches(high_r, [1, 17 * scale, 17 * scale, 1], [1, 14 * scale, 14 * scale, 1],
#                                              [1, 1, 1, 1], "VALID")
#
#     LR_image_patches = tf.reshape(slice_l, [tf.shape(slice_l)[1] * tf.shape(slice_l)[2], 17, 17, channels])
#     HR_image_patches = tf.reshape(slice_h,
#                                   [tf.shape(slice_h)[1] * tf.shape(slice_h)[2], 17 * scale, 17 * scale, channels])
#
#     return tf.data.Dataset.from_tensor_slices((LR_image_patches, HR_image_patches))

# orig with tf.data
# training_batches = tf.placeholder_with_default(tf.constant(32, dtype=tf.int64), shape=[], name="batch_size_input")
# path_inputs = tf.placeholder(tf.string, shape=[None])
# path_dataset = tf.data.Dataset.from_tensor_slices(path_inputs)
# train_dataset = path_dataset.flat_map(lambda x: load_dataset(x, SCALE))
# train_dataset = train_dataset.shuffle(buffer_size=10000)
# train_dataset = train_dataset.batch(training_batches)

# iter = train_dataset.make_initializable_iterator()
# LR, HR = iter.get_next()

# loss, train_op, psnr = ESPCN.ESPCN_model(LR_input=LR, HR_output=HR, scale=SCALE)

##for feed_data
def get_dataset(filenames, scale):
    crop_size_lr = 17
    crop_size_hr = 17 * scale

    x = list()
    y = list()
    for p in filenames:
        image_decoded = cv2.imread(p, 3)
        imgYCC = cv2.cvtColor(image_decoded, cv2.COLOR_BGR2YCrCb) / 255.0
        cropped = imgYCC[0:(imgYCC.shape[0] - (imgYCC.shape[0] % scale)),
                  0:(imgYCC.shape[1] - (imgYCC.shape[1] % scale)), :]
        lr = cv2.resize(cropped, (int(cropped.shape[1] / scale), int(cropped.shape[0] / scale)), interpolation=cv2.INTER_CUBIC)

        hr_y = imgYCC[:, :, 0]
        lr_y = lr[:, :, 0]

        numx = int(lr.shape[0] / crop_size_lr)
        numy = int(lr.shape[1] / crop_size_lr)
        for i in range(0, numx):
            startx = i * crop_size_lr
            endx = (i * crop_size_lr) + crop_size_lr
            startx_hr = i * crop_size_hr
            endx_hr = (i * crop_size_hr) + crop_size_hr
            for j in range(0, numy):
                starty = j * crop_size_lr
                endy = (j * crop_size_lr) + crop_size_lr
                starty_hr = j * crop_size_hr
                endy_hr = (j * crop_size_hr) + crop_size_hr
                crop_lr = lr_y[startx:endx, starty:endy]
                crop_hr = hr_y[startx_hr:endx_hr, starty_hr:endy_hr]
                hr = crop_hr.reshape(((crop_size_hr), (crop_size_hr), 1))
                lr = crop_lr.reshape((crop_size_lr, crop_size_lr, 1))
                x.append(lr)
                y.append(hr)

    #X = np.asarray(x).reshape(-1, 1, 17, 17, 1)
    #Y = np.asarray(y).reshape(-1, 1, 17 * scale, 17 * scale, 1)
    X = x
    Y = y

    return X, Y

##for tf.data
def gen_dataset(filenames, scale):

    crop_size_lr = 17
    crop_size_hr = 17 * scale

    for p in filenames:
        image_decoded = cv2.imread(p.decode(), 3)
        imgYCC = cv2.cvtColor(image_decoded, cv2.COLOR_BGR2YCrCb) / 255.0
        cropped = imgYCC[0:(imgYCC.shape[0] - (imgYCC.shape[0] % scale)),
                  0:(imgYCC.shape[1] - (imgYCC.shape[1] % scale)), :]
        lr = cv2.resize(cropped, (int(cropped.shape[1] / scale), int(cropped.shape[0] / scale)), interpolation=cv2.INTER_CUBIC)

        hr_y = imgYCC[:, :, 0]
        lr_y = lr[:, :, 0]

        numx = int(lr.shape[0] / crop_size_lr)
        numy = int(lr.shape[1] / crop_size_lr)
        for i in range(0, numx):
            startx = i * crop_size_lr
            endx = (i * crop_size_lr) + crop_size_lr
            startx_hr = i * crop_size_hr
            endx_hr = (i * crop_size_hr) + crop_size_hr
            for j in range(0, numy):
                starty = j * crop_size_lr
                endy = (j * crop_size_lr) + crop_size_lr
                starty_hr = j * crop_size_hr
                endy_hr = (j * crop_size_hr) + crop_size_hr
                crop_lr = lr_y[startx:endx, starty:endy]
                crop_hr = hr_y[startx_hr:endx_hr, starty_hr:endy_hr]
                hr = crop_hr.reshape(((crop_size_hr), (crop_size_hr), 1))
                lr = crop_lr.reshape((crop_size_lr, crop_size_lr, 1))
                yield lr, hr


def training(ARGS):
    """
    Start training the ESPCN model.
    """

    print("Starting training...")

    SCALE = ARGS["SCALE"]
    BATCH = int(ARGS["BATCH"])
    print(BATCH)
    EPOCHS = ARGS["EPOCH_NUM"]
    DATA = pathlib.Path(ARGS["TRAINDIR"])
    all_image_paths = list(DATA.glob('*'))
    all_image_paths = [str(path) for path in all_image_paths]

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    print("Loading dataset...")
    X_np, Y_np = get_dataset(all_image_paths, SCALE)

    # ds = tf.data.Dataset.from_generator(
    #     gen_dataset, (tf.float32, tf.float32), (tf.TensorShape([None,None,1]), tf.TensorShape([None,None,1])), args=[all_image_paths,SCALE])
    # train_dataset = ds.batch(BATCH)
    # iter = train_dataset.make_initializable_iterator()
    # LR, HR = iter.get_next()
    # EM = ESPCN.ESPCNmodel(SCALE, LR)
    # model = EM.ESPCN_model_with_tfdata(LR)
    # loss, train_op, psnr = EM.ESPCN_trainable_model(model, HR)


    LR_images = tf.placeholder(tf.float32, [None, None, None, 1], name='images')
    HR_images = tf.placeholder(tf.float32, [None, None, None, 1], name='labels')

    EM = ESPCN.ESPCNmodel(SCALE, LR_images)
    model = EM.ESPCN_model()
    loss, train_op, psnr = EM.ESPCN_trainable_model(model, HR_images)


    with tf.Session(config=config) as sess:
        train_writer = tf.summary.FileWriter('./logs/train', sess.graph)
        sess.run(tf.global_variables_initializer())

        saver = EM.saver
        if not os.path.exists(ARGS["CKPT_dir"]):
            os.makedirs(ARGS["CKPT_dir"])
        else:
            if os.path.isfile(ARGS["CKPT"] + ".meta"):
                saver.restore(sess, tf.train.latest_checkpoint(ARGS["CKPT_dir"]))
                print("Loaded checkpoint.")
            else:
                print("Previous checkpoint does not exists.")


        ##training with old feed_dict method
        for e in range(EPOCHS):
            count = 0
            num_of_batches = (len(X_np)//BATCH)-1
            for j in range(0, num_of_batches-1):
                batch_x = X_np[(j*BATCH):(j*BATCH)+BATCH]
                batch_y = Y_np[(j*BATCH):(j*BATCH)+BATCH]
                l, t, ps = sess.run([loss, train_op, psnr], feed_dict={EM.LR_input: batch_x, HR_images: batch_y})
                count = count + 1

                if count % 50 == 0:
                    print("Data count:", '%04d' % (count + 1), "Epoch no:", '%04d' % (e + 1), "loss:", "{:.9f}".format(l))
                    save_path = saver.save(sess, ARGS["CKPT"])

        # sess.run(iter.initializer)
        ##training with tf.data method
        # saver = tf.train.Saver()
        # for e in range(EPOCHS):
        #     count = 0
        #     try:
        #         while True:
        #             count = count + 1
        #             l, t, ps = sess.run([loss, train_op, psnr])
        #             if count % 1 == 0:
        #                 print("Data count:", '%04d' % (count + 1), "Epoch no:", '%04d' % (e + 1), "loss:",
        #                         "{:.9f}".format(l))
        #                 save_path = saver.save(sess, ARGS["CKPT"])
        #     except tf.errors.OutOfRangeError:
        #         pass

        train_writer.close()

def test(ARGS):
    print("Not implemented yet.")
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    SCALE = ARGS['SCALE']

    #Read test image
    img = cv2.imread('Test/t3.png', 3)

    #Make it divisible by scale factor
    cropped = img[0:(img.shape[0] - (img.shape[0] % SCALE)), 0:(img.shape[1] - (img.shape[1] % SCALE)), :]

    floatimg = cropped.astype(np.float32) / 255.0

    #Convert to YCbCr color space
    imgYCbCr = cv2.cvtColor(floatimg, cv2.COLOR_BGR2YCrCb)

    imgY = imgYCbCr[:,:,0]

    LR_input_ = imgY.reshape(1, imgY.shape[0], imgY.shape[1], 1)

    print("Start running tests on the model")
    graph = tf.get_default_graph()
    with graph.as_default():
        with tf.Session(config=config) as sess:
            EM = ESPCN.ESPCNmodel(SCALE,tf.placeholder(tf.float32, [None, None, None, 1], name='images'))
            model = EM.ESPCN_model()
            ### Restore checkpoint
            EM.load_checkpoint(sess, ckpt_dir=ARGS["CKPT_dir"])
            output = sess.run(model, feed_dict={EM.LR_input: LR_input_})

            #load the model with tf.data generator
            #ckpt_name = ARGS["CKPT"] + ".meta"
            #saver = tf.train.import_meta_graph(ckpt_name)
            #saver.restore(sess, tf.train.latest_checkpoint(ARGS["CKPT_dir"]))
            #LR_tensor = graph.get_tensor_by_name("IteratorGetNext:0")
            #HR_tensor = graph.get_tensor_by_name("NHWC_output:0")
            #output = sess.run(HR_tensor, feed_dict={LR_tensor: LR_input_})

            Y = output[0]
            Cr = np.expand_dims(cv2.resize(imgYCbCr[:,:,1], None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_CUBIC), axis=2)
            Cb = np.expand_dims(cv2.resize(imgYCbCr[:,:,2], None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_CUBIC), axis=2)

            HR_image = (cv2.cvtColor(np.concatenate((Y, Cr, Cb), axis=2), cv2.COLOR_YCrCb2BGR))

            cv2.imshow('LR image', cropped)
            cv2.imshow('HR image', HR_image)
            cv2.imshow('HR image grey', output[0])
            cv2.waitKey(0)


def export(ARGS):
    print("Exporting model.")

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    SCALE = ARGS['SCALE']

    print("Start running tests on the model")
    graph = tf.get_default_graph()
    with graph.as_default():
        with tf.Session(config=config) as sess:
            EM = ESPCN.ESPCNmodel(SCALE,tf.placeholder(tf.float32, [None, None, None, 1], name='images'))
            model = EM.ESPCN_model()
            EM.load_checkpoint(sess, ckpt_dir=ARGS["CKPT_dir"])

            graph_def = sess.graph.as_graph_def()
            graph_def = tf.graph_util.convert_variables_to_constants(sess, graph_def, ['NCHW_output'])
            # graph_def = optimize_for_inference_lib.optimize_for_inference(graph_def, ["IteratorGetNext"],
            graph_def = optimize_for_inference_lib.optimize_for_inference(graph_def, ["images"],
                                                                          ["NCHW_output"],  # ["NHWC_output"],
                                                                          tf.float32.as_datatype_enum)
            # graph_def = TransformGraph(graph_def, ["IteratorGetNext"], ["NCHW_output"], ["sort_by_execution_order"])
            graph_def = TransformGraph(graph_def, ["images"], ["NCHW_output"], ["sort_by_execution_order"])

            with tf.gfile.FastGFile('frozen_inference_graph_opt.pb', 'wb') as f:
                f.write(graph_def.SerializeToString())

            tf.train.write_graph(graph_def, ".", 'train.pbtxt')
