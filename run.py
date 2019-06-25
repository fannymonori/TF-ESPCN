import tensorflow as tf
import pathlib
import cv2
import numpy as np
import os
import math

from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib
from tensorflow.tools.graph_transforms import TransformGraph

import ESPCN
import utils


def training(ARGS):
    """
    Start training the ESPCN model.
    """

    print("\nStarting training...\n")

    SCALE = ARGS["SCALE"]
    BATCH = 32
    EPOCHS = ARGS["EPOCH_NUM"]
    DATA = pathlib.Path(ARGS["TRAINDIR"])
    LRATE = ARGS["LRATE"]

    all_image_paths = list(DATA.glob('*'))
    all_image_paths = [str(path) for path in all_image_paths]

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    ## Creating a tf.data pipeline, so it can train on a large dataset without storing it in the memory.
    ds = tf.data.Dataset.from_generator(
        utils.gen_dataset, (tf.float32, tf.float32), (tf.TensorShape([None, None, 1]), tf.TensorShape([None, None, 1])),
        args=[all_image_paths, SCALE])
    train_dataset = ds.batch(BATCH)
    train_dataset = train_dataset.shuffle(10000)
    iter = train_dataset.make_initializable_iterator()
    LR, HR = iter.get_next()
    EM = ESPCN.ESPCN(input=LR, scale=SCALE, learning_rate=LRATE)
    model = EM.ESPCN_model()
    loss, train_op, psnr = EM.ESPCN_trainable_model(model, HR)

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        train_writer = tf.summary.FileWriter('./logs/train', sess.graph)

        saver = EM.saver
        if not os.path.exists(ARGS["CKPT_dir"]):
            os.makedirs(ARGS["CKPT_dir"])
        else:
            if os.path.isfile(ARGS["CKPT"] + ".meta"):
                saver.restore(sess, tf.train.latest_checkpoint(ARGS["CKPT_dir"]))
                print("Loaded checkpoint.")
            else:
                print("Previous checkpoint does not exists.")

        # training with tf.data method
        saver = tf.train.Saver()
        for e in range(EPOCHS):
            sess.run(iter.initializer)
            count = 0
            train_loss, train_psnr = 0.0, 0.0
            while True:
                try:
                    count = count + 1
                    l, t, ps = sess.run([loss, train_op, psnr])
                    train_loss += l
                    train_psnr += (np.mean(np.asarray(ps)))

                    if count % 100 == 0:
                        print("Data num:", '%d' % (count * 32), "Epoch no:", '%04d' % (e + 1), "loss:", "{:.9f}".format(l),
                              "epoch loss:", "{:.9f}".format(train_loss / (count)),
                              "epoch psnr:", "{:.9f}".format(train_psnr / (count)))
                        saver.save(sess, ARGS["CKPT"])

                except tf.errors.OutOfRangeError:
                    break
            print("END OF EPOCH - Epoch no:", '%04d' % (e + 1), "loss:", "{:.9f}".format(l),
                  "epoch loss:", "{:.9f}".format(train_loss / (count)),
                  "epoch psnr:", "{:.9f}".format(train_psnr / (count)))
            saver.save(sess, ARGS["CKPT"])

        train_writer.close()


def test(ARGS):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    SCALE = ARGS['SCALE']

    path = ARGS["TESTIMG"]

    fullimg = cv2.imread(path, 3)
    width = fullimg.shape[0]
    height = fullimg.shape[1]

    cropped = fullimg[0:(width - (width % SCALE)), 0:(height - (height % SCALE)), :]
    img = cv2.resize(cropped, None, fx=1. / SCALE, fy=1. / SCALE, interpolation=cv2.INTER_CUBIC)
    floatimg = img.astype(np.float32) / 255.0

    # Convert to YCbCr color space
    imgYCbCr = cv2.cvtColor(floatimg, cv2.COLOR_BGR2YCrCb)
    imgY = imgYCbCr[:, :, 0]

    LR_input_ = imgY.reshape(1, imgY.shape[0], imgY.shape[1], 1)

    # with tf.gfile.GFile("nchw_frozen_ESPCN_graph_x2.pb", 'rb') as f:
    #     graph_def = tf.GraphDef()
    #     graph_def.ParseFromString(f.read())
    #     g = tf.import_graph_def(graph_def)
    #     # output_tensor = sess.graph.get_tensor_by_name("IteratorGetNext")
    #
    # sess = tf.Session(graph=g,config=config)
    #
    # LR_tensor = sess.graph.get_tensor_by_name("import/IteratorGetNext:0")
    # inp = cv2.cvtColor((cropped.astype(np.float32) / 255.0), cv2.COLOR_BGR2YCrCb)[:,:,0].reshape(1, cropped.shape[0], cropped.shape[1], 1)
    # output = sess.run(sess.graph.get_tensor_by_name("import/NCHW_output:0"), feed_dict={LR_tensor: inp})
    # Y = output[0][0]
    # print(Y.shape)
    # cv2.imshow('Bicubic HR image', Y)
    # cv2.waitKey(0)

    with tf.Session(config=config) as sess:
        print("\nStart running tests on the model\n")
        # #load the model with tf.data generator
        ckpt_name = ARGS["CKPT"] + ".meta"
        saver = tf.train.import_meta_graph(ckpt_name)
        saver.restore(sess, tf.train.latest_checkpoint(ARGS["CKPT_dir"]))
        graph_def = sess.graph
        LR_tensor = graph_def.get_tensor_by_name("IteratorGetNext:0")
        HR_tensor = graph_def.get_tensor_by_name("NHWC_output:0")

        output = sess.run(HR_tensor, feed_dict={LR_tensor: LR_input_})

        Y = output[0]
        Cr = np.expand_dims(cv2.resize(imgYCbCr[:, :, 1], None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_CUBIC),
                            axis=2)
        Cb = np.expand_dims(cv2.resize(imgYCbCr[:, :, 2], None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_CUBIC),
                            axis=2)

        HR_image_YCrCb = np.concatenate((Y, Cr, Cb), axis=2)
        HR_image = ((cv2.cvtColor(HR_image_YCrCb, cv2.COLOR_YCrCb2BGR)) * 255.0).clip(min=0, max=255)
        HR_image = (HR_image).astype(np.uint8)

        bicubic_image = cv2.resize(img, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_CUBIC)

        print(np.amax(Y), np.amax(LR_input_))
        print("PSNR of ESPCN generated image: ", utils.PSNR(cropped, HR_image))
        print("PSNR of bicubic interpolated image: ", utils.PSNR(cropped, bicubic_image))

        cv2.imshow('Original image', fullimg)
        cv2.imshow('HR image', HR_image)
        cv2.imshow('Bicubic HR image', bicubic_image)
        cv2.waitKey(0)

def export(ARGS):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    SCALE = ARGS['SCALE']

    print("\nStart exporting the model...\n")
    with tf.Session(config=config) as sess:
        ckpt_name = ARGS["CKPT"] + ".meta"
        saver = tf.train.import_meta_graph(ckpt_name)
        saver.restore(sess, tf.train.latest_checkpoint(ARGS["CKPT_dir"]))

        graph_def = sess.graph.as_graph_def()
        graph_def = tf.graph_util.convert_variables_to_constants(sess, graph_def, ['NHWC_output'])
        graph_def = optimize_for_inference_lib.optimize_for_inference(graph_def, ["IteratorGetNext"],
                                                                      ["NHWC_output"],
                                                                      tf.float32.as_datatype_enum)
        graph_def = TransformGraph(graph_def, ["IteratorGetNext"], ["NHWC_output"], ["sort_by_execution_order"])

        filename = 'frozen_ESPCN_graph_x' + str(SCALE) + '.pb'
        with tf.gfile.FastGFile(filename, 'wb') as f:
            f.write(graph_def.SerializeToString())

        tf.train.write_graph(graph_def, ".", 'train.pbtxt')

        #SAVE NCHW
        graph_def = sess.graph.as_graph_def()
        graph_def = tf.graph_util.convert_variables_to_constants(sess, graph_def, ['NCHW_output'])
        graph_def = optimize_for_inference_lib.optimize_for_inference(graph_def, ["IteratorGetNext"],
                                                                      ["NCHW_output"],
                                                                      tf.float32.as_datatype_enum)
        graph_def = TransformGraph(graph_def, ["IteratorGetNext"], ["NCHW_output"], ["sort_by_execution_order"])
        filename = 'nchw_frozen_ESPCN_graph_x' + str(SCALE) + '.pb'
        with tf.gfile.FastGFile(filename, 'wb') as f:
            f.write(graph_def.SerializeToString())

        tf.train.write_graph(graph_def, ".", 'nchw_train.pbtxt')

    print("\nExporting done!\n")
