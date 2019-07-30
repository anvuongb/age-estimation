import argparse
from pathlib import Path
import numpy as np
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.optimizers import SGD, Adam
from generator import FaceGenerator, FaceValGenerator
from model import get_model, age_mae, mean_variance_loss


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
    parser.add_argument("--meta-test-csv", type=str, required=True,
                        help="path to the test dataset csv")
    parser.add_argument("--output-dir", type=str, default="checkpoints",
                        help="checkpoint dir")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="batch size")
    parser.add_argument("--nb-epochs", type=int, default=30,
                        help="number of epochs")
    parser.add_argument("--lr", type=float, default=0.1,
                        help="learning rate")
    parser.add_argument("--opt", type=str, default="sgd",
                        help="optimizer name; 'sgd' or 'adam'")
    parser.add_argument("--model-name", type=str, default="ResNet50",
                        help="model name: ResNet50 or InceptionResNetV2 or InceptionV3")
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
    else:
        raise ValueError("optimizer name should be 'sgd' or 'adam'")


def main():
    args = get_args()
    meta_train_csv = args.meta_train_csv
    meta_test_csv = args.meta_test_csv
    model_name = args.model_name
    batch_size = args.batch_size
    nb_epochs = args.nb_epochs
    lr = args.lr
    opt_name = args.opt

    if model_name == "ResNet50":
        image_size = 224
    elif model_name == "InceptionResNetV2":
        image_size = 299

    train_gen = FaceGenerator(meta_train_csv, batch_size=batch_size, image_size=image_size)
    val_gen = FaceValGenerator(meta_test_csv, batch_size=batch_size, image_size=image_size)
    model = get_model(model_name=model_name)
    opt = get_optimizer(opt_name, lr)
    model.compile(optimizer=opt, loss=mean_variance_loss, metrics=[age_mae])
    model.summary()
    output_dir = Path(__file__).resolve().parent.joinpath(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    callbacks = [LearningRateScheduler(schedule=Schedule(nb_epochs, initial_lr=lr)),
                 ModelCheckpoint(str(output_dir) + "/weights.{epoch:03d}-{val_loss:.3f}-{val_age_mae:.3f}.hdf5",
                                 monitor="val_age_mae",
                                 verbose=1,
                                 save_best_only=True,
                                 mode="min")
                 ]

    hist = model.fit_generator(generator=train_gen,
                               epochs=nb_epochs,
                               validation_data=val_gen,
                               verbose=1,
                               callbacks=callbacks)

    np.savez(str(output_dir.joinpath("history.npz")), history=hist.history)


if __name__ == '__main__':
    main()
