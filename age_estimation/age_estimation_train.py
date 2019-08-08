import argparse
from pathlib import Path
import numpy as np
import os

# os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import SGD, Adam, Adadelta
import tensorflow.keras.backend as K
import tensorflow as tf
from generator import FaceGenerator, FaceValGenerator
from model import get_model, age_mae, mean_variance_loss
from datetime import datetime



###########################################################
# GENERATOR WILL REQUIRE N CSV WITH THE FOLLOWING FORMAT  #
# img_path,age                                            #
# abs_path_1,age1                                         #
# abs_path_2,age2                                         #
###########################################################

def get_args():
    parser = argparse.ArgumentParser(description="This script trains the CNN model for age estimation.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--meta-train-csv", type=str, required=True,
                        help="path to the train dataset csv")
    parser.add_argument("--meta-val-csv", type=str, required=True,
                        help="path to the test dataset csv")
    parser.add_argument("--output-dir", type=str, default="training_output",
                        help="output directory, both history and checkpoints")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="batch size")
    parser.add_argument("--nb-epochs", type=int, default=30,
                        help="number of epochs")
    parser.add_argument("--lr", type=float, default=0.1,
                        help="learning rate")
    parser.add_argument("--opt", type=str, default="sgd",
                        help="optimizer name; 'sgd' or 'adam' or 'adadelta")
    parser.add_argument("--model-name", type=str, default="ResNet50",
                        help="model name: ResNet50 or InceptionResNetV2 or InceptionV3 or SEInceptionV3")
    parser.add_argument("--weight-file", type=str, 
                        help="continue to train from a pretrained model")
    parser.add_argument("--log-freq", type=int, default=2000,
                        help="tensorboard log every x number of instances")
    parser.add_argument("--gpu", type=int, default=0,
                        help="gpu to train on") 
    parser.add_argument("--num-workers", type=int, default=1,
                        help="number of cpu threads for generating batches")
    parser.add_argument("--queue-size", type=int, default=10,
                        help="number of batches prepared in advance")
    parser.add_argument("--gpu-frac", type=float, default=1.0,
                        help="fraction of gpu memory to allocate")
    parser.add_argument("--provider", type=str, default="nvidia",
                        help="use nvidia or amd gpu")
    parser.add_argument("--fp16", type=int, default=0,
                        help="use fp16 mode provided by tensorflow")                                                     
    args = parser.parse_args()
    return args


class Schedule:
    def __init__(self, nb_epochs, initial_lr):
        self.epochs = nb_epochs
        self.initial_lr = initial_lr

    def __call__(self, epoch_idx):
        if epoch_idx < self.epochs * 0.25:
            return self.initial_lr
        elif epoch_idx < self.epochs * 0.50:
            return self.initial_lr * 0.2
        elif epoch_idx < self.epochs * 0.75:
            return self.initial_lr * 0.04
        return self.initial_lr * 0.008


def get_optimizer(opt_name, lr):
    if opt_name == "sgd":
        return SGD(lr=lr, momentum=0.9, nesterov=True)
    elif opt_name == "adam":
        return Adam(lr=lr)
    elif opt_name == "adadelta":
        return Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
    else:
        raise ValueError("optimizer name should be 'sgd' or 'adam' or 'adadelta")


def main():
    args = get_args()

    if args.fp16==1:
        os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1' 

    if args.provider == "amd":
        os.environ["HIP_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
        os.environ["HIP_VISIBLE_DEVICES"]="{}".format(args.gpu)
    elif args.provider == "nvidia":
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"]="{}".format(args.gpu)

    print("Training will be done on {} GPU {}".format(args.provider, args.gpu))

    meta_train_csv = args.meta_train_csv
    meta_val_csv = args.meta_val_csv
    model_name = args.model_name
    batch_size = args.batch_size
    nb_epochs = args.nb_epochs
    lr = args.lr
    opt_name = args.opt
    output_dir = args.output_dir
    log_freq = args.log_freq
    num_workers = args.num_workers
    max_queue_size = args.queue_size

    weight_file = args.weight_file

    # set session GPU memory
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = args.gpu_frac
    K.set_session(tf.Session(config=config))

    # Initialize model input size
    if model_name == "ResNet50":
        image_size = 224
    elif model_name == "InceptionResNetV2" or model_name == "InceptionV3" or model_name =="SEInceptionV3":
        image_size = 299

    # Initialize generator
    train_gen = FaceGenerator(meta_train_csv, batch_size=batch_size, image_size=image_size)
    val_gen = FaceValGenerator(meta_val_csv, batch_size=batch_size, image_size=image_size)

    # Get model
    model = get_model(model_name=model_name)

    if weight_file is not None:
        model.load_weights(weight_file)

    opt = get_optimizer(opt_name, lr)
    if args.fp16 == 1:
        opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt) 
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=[age_mae])
    model.summary()

    # Create output directory
    output_dir = Path(__file__).resolve().parent.joinpath(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    weights_dir = output_dir.joinpath("weights")
    weights_dir = Path(__file__).resolve().parent.joinpath(weights_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)

    # Initialize tensorboard logging callback
    dt_now = datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = output_dir.joinpath("logs/scalars/{}-{}-{}".format(model_name, opt_name, dt_now))
    logdir = Path(__file__).resolve().parent.joinpath(logdir)
    logdir.mkdir(parents=True, exist_ok=True)

    tensorboard_callback = TensorBoard(log_dir=str(logdir), update_freq=log_freq)

    # Initialize callbacks
    if opt_name is not 'adadelta':
        callbacks = [LearningRateScheduler(schedule=Schedule(nb_epochs, initial_lr=lr)),
                    ModelCheckpoint(str(weights_dir) + "/{}.".format(model_name) + "weights.{epoch:03d}-{val_loss:.3f}-{val_age_mae:.3f}" + "-{}.hdf5".format(dt_now),
                                    monitor="val_age_mae",
                                    verbose=1,
                                    save_best_only=True,
                                    mode="min"),
                    tensorboard_callback
                    ]
    else:
        callbacks = [ModelCheckpoint(str(weights_dir) + "/{}.".format(model_name) + "weights.{epoch:03d}-{val_loss:.3f}-{val_age_mae:.3f}" + "-{}.hdf5".format(dt_now),
                                    monitor="val_age_mae",
                                    verbose=1,
                                    save_best_only=True,
                                    mode="min"),
                    tensorboard_callback
                    ]

    if num_workers > 1:
        model.fit_generator(generator=train_gen,
                               epochs=nb_epochs,
                               validation_data=val_gen,
                               verbose=1,
                               callbacks=callbacks,
                               use_multiprocessing=True,
                               workers=num_workers,
                               max_queue_size=max_queue_size)
    elif num_workers==1:
        model.fit_generator(generator=train_gen,
                               epochs=nb_epochs,
                               validation_data=val_gen,
                               verbose=1,
                               callbacks=callbacks,
                               max_queue_size=max_queue_size)
    

    # np.savez(str(output_dir.joinpath("history.npz")), history=hist.history)


if __name__ == '__main__':
    main()
