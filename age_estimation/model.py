import better_exceptions
from keras.applications import ResNet50, InceptionResNetV2, InceptionV3
from keras.layers import Dense
from keras.models import Model
from keras import backend as K
import numpy as np

def age_mae(y_true, y_pred):
    true_age = K.sum(y_true * K.arange(0, 70, dtype="float32"), axis=-1)
    pred_age = K.sum(y_pred * K.arange(0, 70, dtype="float32"), axis=-1)
    mae = K.mean(K.abs(true_age - pred_age))
    return mae


def get_model(model_name="ResNet50"):
    base_model = None

    if model_name == "ResNet50":
        base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3), pooling="avg")
    elif model_name == "InceptionResNetV2":
        base_model = InceptionResNetV2(include_top=False, weights='imagenet', input_shape=(299, 299, 3), pooling="avg")
    elif model_name == "InceptionV3":
        base_model = InceptionV3(include_top=False, weights='imagenet', input_shape=(299, 299, 3), pooling="avg")

    prediction = Dense(units=70, kernel_initializer="he_normal", use_bias=False, activation="softmax",
                       name="pred_age")(base_model.output)

    model = Model(inputs=base_model.input, outputs=prediction)

    return model

def mean_variance_loss(y_true, y_pred):
    n_bins = 70
    ages = K.constant(np.arange(0, n_bins), dtype=float)

    mean_age_true = K.sum(y_true * ages, axis=-1)
    mean_age_pred = K.sum(y_pred * ages, axis=-1)

    mean_loss = K.mean(K.square(mean_age_pred - mean_age_true), axis=-1)
    softmax_loss = K.sum(K.categorical_crossentropy(y_true, y_pred, axis=-1), axis=-1)
    variance_loss = K.mean(K.sum(K.square(y_pred * ages - ages) * y_pred, axis=-1), axis=-1)
    
    total_loss = mean_loss + softmax_loss + variance_loss
    return total_loss


def main():
    model = get_model("ResNet50")
    model.summary()


if __name__ == '__main__':
    main()
