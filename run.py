import tensorflow as tf
import pathlib
import ESPCN
import cv2
import numpy as np
import os

from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib
from tensorflow.tools.graph_transforms import TransformGraph


def load_dataset(path, scale):
    """
    Load the jpeg dataset for training.

    Parameters
    ----------
    path: list of strings
        Path of images
    scale: int
        Super-resolution scale
    channels: int
        Number of channels used for training
    """

    scale_factor = tf.constant(scale)
    channels = 1
    image = tf.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)

    image = tf.cast(image, tf.float32)
    R, G, B = tf.unstack(image, 3, axis=2)

    y = R * 0.299 + G * 0.587 + B * 0.114
    image_preprocessed = tf.reshape(y, (tf.shape(image)[0], tf.shape(image)[1], 1)) / 255

    X = tf.dtypes.cast((tf.shape(image_preprocessed)[0] / scale_factor), dtype=tf.int32) * scale_factor
    Y = tf.dtypes.cast((tf.shape(image_preprocessed)[1] / scale_factor), dtype=tf.int32) * scale_factor
    high = tf.image.crop_to_bounding_box(image_preprocessed, 0, 0, X, Y)

    imgshape = tf.shape(high)
    size = [imgshape[0] / scale_factor, imgshape[1] / scale_factor]
    low = tf.image.resize_images(high, size=size, method=tf.image.ResizeMethod.BILINEAR)

    hshape = tf.shape(high)
    lshape = tf.shape(low)

    high_r = tf.reshape(high, [1, hshape[0], hshape[1], channels])
    low_r = tf.reshape(low, [1, lshape[0], lshape[1], channels])

    slice_l = tf.image.extract_image_patches(low_r, [1, 17, 17, 1], [1, 17, 17, 1], [1, 1, 1, 1], "VALID")
    slice_h = tf.image.extract_image_patches(high_r, [1, 17 * scale, 17 * scale, 1], [1, 17 * scale, 17 * scale, 1],
                                             [1, 1, 1, 1], "VALID")

    LR_image_patches = tf.reshape(slice_l, [tf.shape(slice_l)[1] * tf.shape(slice_l)[2], 17, 17, channels])
    HR_image_patches = tf.reshape(slice_h, [tf.shape(slice_h)[1] * tf.shape(slice_h)[2], 17 * scale, 17 * scale, channels])

    return tf.data.Dataset.from_tensor_slices((LR_image_patches, HR_image_patches))


def training(ARGS):
    """
    Start training the ESPCN model.
    """

    print("Starting training...")

    SCALE = ARGS["SCALE"]
    BATCH = int(ARGS["BATCH"])
    EPOCHS = ARGS["EPOCH_NUM"]
    DATA = pathlib.Path(ARGS["TRAINDIR"])
    all_image_paths = list(DATA.glob('*'))
    all_image_paths = [str(path) for path in all_image_paths]

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    training_batches = tf.placeholder_with_default(tf.constant(32, dtype=tf.int64), shape=[], name="batch_size_input")
    path_inputs = tf.placeholder(tf.string, shape=[None])

    path_dataset = tf.data.Dataset.from_tensor_slices(path_inputs)
    train_dataset = path_dataset.flat_map(lambda x: load_dataset(x, SCALE))

    train_dataset = train_dataset.shuffle(buffer_size=10000)
    train_dataset = train_dataset.batch(training_batches)

    iter = train_dataset.make_initializable_iterator()

    LR, HR = iter.get_next()
    next = iter.get_next()
    loss, train_op, psnr = ESPCN.ESPCN_model(LR_input=LR, HR_output=HR, scale=SCALE)

    with tf.Session(config=config) as sess:
        # merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter('./logs/train', sess.graph)
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        if not os.path.exists(ARGS["CKPT_dir"]):
            os.makedirs(ARGS["CKPT_dir"])
        else:
            if os.path.isfile(ARGS["CKPT"] + ".meta"):
                saver.restore(sess, tf.train.latest_checkpoint(ARGS["CKPT_dir"]))
                print("Loaded checkpoint.")
            else:
                print("Previous checkpoint does not exists.")

        for e in range(EPOCHS):
            sess.run(iter.initializer, feed_dict={path_inputs: all_image_paths, training_batches: BATCH})
            count = 0
            sess.run(next)
            try:
                while True:
                    l, t, ps = sess.run([loss, train_op, psnr])
                    count = count + 1
                    if count % 200 == 0:
                        print("Data count:", '%04d' % (count + 1), "Epoch no:", '%04d' % (e + 1), "loss:","{:.9f}".format(l))
                    if count % 1000 == 0:
                        save_path = saver.save(sess, ARGS["CKPT"])
                        print("Model saved in path: %s" % save_path)
            except tf.errors.OutOfRangeError:
                pass

        train_writer.close()

# def split(image,nrows,ncols):
#     n, m = image.shape
#     return (image.reshape(n // nrows, nrows, -1, ncols).swapaxes(1, 2).reshape(-1, nrows, ncols))

def test(ARGS):
    print("Not implemented yet.")
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    #
    # scale = ARGS['SCALE']
    #
    # lr_size = 17
    # hr_size = 17 * scale
    #
    # img = cv2.imread('Test/t1.png', 3)
    # imgYCC = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    #
    # img_y_norm = imgYCC / 255
    #
    # numx = int(img.shape[0] / 17) + 1
    # numy = int(img.shape[1] / 17) + 1
    # ### create tiles from image
    # tiles = np.zeros((numx, numy, 17, 17, 3))
    # for i in range(0, numx):
    #     startx = i * 17
    #     endx = (i * 17) + 17
    #     for j in range(0, numy):
    #         starty = j * 17
    #         endy = (j * 17) + 17
    #         if i == numx - 1 and j == numy - 1:
    #             endy = img_y_norm.shape[1]
    #             endx = img_y_norm.shape[0]
    #             c = img_y_norm[(endx - 17):endx, (endy - 17):endy, :]
    #         elif i == numx - 1:
    #             endx = img_y_norm.shape[0]
    #             c = img_y_norm[(endx - 17):endx, starty:endy, :]
    #         elif j == numy - 1:
    #             endy = img_y_norm.shape[1]
    #             c = img_y_norm[startx:endx, (endy - 17):endy, :]
    #         else:
    #             c = img_y_norm[startx:endx, starty:endy, :]
    #         tiles[i, j, :, :, :] = c
    #
    # print(tiles.shape)
    #
    # cropped = img_y_norm[0:(img.shape[0] - (img.shape[0] % 17)), 0:(img.shape[1] - (img.shape[1] % 17)),:]
    #
    # #[split(x,17,17) for x in np.split(img_y_norm,2)]
    # sp = [split(cropped[:,:,x],17,17) for x in range(0,3)]
    # tiles = np.asarray(sp).reshape(-1, 17, 17, 3)
    #
    # tiles_num = numx*numy
    #
    # print("Start running tests on the model")
    # graph = tf.get_default_graph()
    # with graph.as_default():
    #     with tf.Session(config=config) as sess:
    #
    #         ### Restore checkpoint
    #         ckpt_name = ARGS["CKPT"] + ".meta"
    #         print(ARGS["CKPT_dir"])
    #         saver = tf.train.import_meta_graph(ckpt_name)
    #         saver.restore(sess, tf.train.latest_checkpoint(ARGS["CKPT_dir"]))
    #
    #         ### Get input and output nodes
    #         LR_tensor = graph.get_tensor_by_name("IteratorGetNext:0")
    #         HR_tensor = graph.get_tensor_by_name("NHWC_output:0")
    #
    #         #reshaped = tiles.reshape((numx * numy, 17, 17, 3))
    #         reshaped = tiles
    #
    #         LR_input = reshaped[:,:,:,0].reshape(reshaped.shape[0],reshaped.shape[1],reshaped.shape[2],1)
    #
    #         ### Run inference
    #         opt = sess.run(HR_tensor, feed_dict={LR_tensor: LR_input})
    #         print(opt.shape)
    #         print(reshaped.shape)
    #
    #         ### Get 100th tile
    #         #img = opt[100].reshape((hr_size, hr_size, 1))
    #
    #         rescaled = np.zeros((tiles_num, hr_size, hr_size, 3))
    #
    #         #HR_img_normed = np.zeros((17*scale*numx, 17*scale*numy, 3))
    #         for i in range(0,tiles_num):
    #             resized_cb = cv2.resize(reshaped[i, :, : ,1], None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    #             resized_cr = cv2.resize(reshaped[i, :, :, 2], None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    #             rescaled[i, :, :, 0] = opt[i].reshape((hr_size,hr_size))
    #             rescaled[i, :, :, 1] = resized_cb
    #             rescaled[i, :, :, 2] = resized_cr
    #
    #         temp = rescaled.reshape((tiles_num,hr_size,hr_size,3))
    #         HR_img_normed = temp.reshape((numx*hr_size, numy*hr_size, 3))
    #
    #         #HR_img_normed = np.zeros((hr_size, hr_size, 3))
    #         #HR_img_normed[:, :, 0] = img.reshape((hr_size,hr_size))
    #         #HR_img_normed[:, :, 1] = resized_cb
    #         #HR_img_normed[:, :, 2] = resized_cr
    #         print(HR_img_normed.shape)
    #         HR_img = cv2.cvtColor((HR_img_normed * 255).astype(np.uint8), cv2.COLOR_YCrCb2BGR)
    #
    #         cv2.namedWindow('HR patch', cv2.WINDOW_NORMAL)
    #         cv2.imshow('HR patch', HR_img)
    #         cv2.waitKey(0)


def export(ARGS):
    print("Exporting model.")

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    print("Start running tests on the model")
    graph = tf.get_default_graph()
    with graph.as_default():
        with tf.Session(config=config) as sess:
            ckpt_name = ARGS["CKPT"] + ".meta"
            saver = tf.train.import_meta_graph(ckpt_name)
            saver.restore(sess, tf.train.latest_checkpoint(ARGS["CKPT_dir"]))
            graph_def = sess.graph.as_graph_def()
            graph_def = tf.graph_util.convert_variables_to_constants(sess, graph_def, ['NCHW_output'])
            graph_def = optimize_for_inference_lib.optimize_for_inference(graph_def, ["IteratorGetNext"],
                                                                          ["NCHW_output"],  # ["NHWC_output"],
                                                                          tf.float32.as_datatype_enum)
            graph_def = TransformGraph(graph_def, ["IteratorGetNext"], ["NCHW_output"], ["sort_by_execution_order"])

            with tf.gfile.FastGFile('frozen_inference_graph_opt.pb', 'wb') as f:
                f.write(graph_def.SerializeToString())

            tf.train.write_graph(graph_def, ".", 'train.pbtxt')
