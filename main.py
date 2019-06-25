import run
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', help='Start training the model.', action="store_true")
    parser.add_argument('--test', help='Run tests on the model', action="store_true")
    parser.add_argument('--export', help='Export model to .pb and .pbtxt format', action="store_true")
    parser.add_argument('--traindir', help='Training images directory', default="./Test")
    parser.add_argument('--testimg', help='Test image', default="./Test/t1.png")
    parser.add_argument('--scale', help='Scaling factor of the model', default=2)
    parser.add_argument('--epoch', help='Number of epochs during training', default=100)
    parser.add_argument('--lr', help='Sets the learning rate', default=0.01)
    args = parser.parse_args()

    ARGS = dict()
    ARGS["SCALE"] = int(args.scale)

    main_ckpt_dir = "./checkpoints"
    if not os.path.exists(main_ckpt_dir):
        os.makedirs(main_ckpt_dir)

    ARGS["CKPT_dir"] = main_ckpt_dir + "/checkpoint" + "_sc" + str(args.scale)
    ARGS["CKPT"] = ARGS["CKPT_dir"] + "/ESPCN_ckpt_sc" + str(args.scale)
    ARGS["TRAINDIR"] = args.traindir
    ARGS["EPOCH_NUM"] = int(args.epoch)
    ARGS["TESTIMG"] = args.testimg
    ARGS["LRATE"] = float(args.lr)

    if args.train:
        run.training(ARGS)
    elif args.test:
        run.test(ARGS)
    elif args.export:
        run.export(ARGS)
