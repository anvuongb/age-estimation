import better_exceptions
from tensorflow.keras.applications import ResNet50, InceptionResNetV2, InceptionV3
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
import numpy as np

from SE_InceptionV3_keras import SEInceptionV3

# CHANGE NUM_AGE_BINS ACCORDINGLY

NUM_AGE_BINS = 70
MEAN_VARIANCE_ALPHA = 0.7

def age_mae(y_true, y_pred):
    true_age = K.sum(y_true * K.arange(0, NUM_AGE_BINS, dtype="float32"), axis=-1)
    pred_age = K.sum(y_pred * K.arange(0, NUM_AGE_BINS, dtype="float32"), axis=-1)
    mae = K.mean(K.abs(true_age - pred_age))
    return mae

def get_model(model_name="ResNet50", n_bins=NUM_AGE_BINS, weights='imagenet'):
    base_model = None

    if model_name == "ResNet50":
        base_model = ResNet50(include_top=False, weights=weights, input_shape=(224, 224, 3), pooling="avg")
    elif model_name == "InceptionResNetV2":
        base_model = InceptionResNetV2(include_top=False, weights=weights, input_shape=(299, 299, 3), pooling="avg")
    elif model_name == "InceptionV3":
        base_model = InceptionV3(include_top=False, weights=weights, input_shape=(299, 299, 3), pooling="avg")
    elif model_name == "SEInceptionV3":
        base_model = SEInceptionV3(include_top=False, weights=weights, input_shape=(299, 299, 3), pooling="avg")

    prediction = Dense(n_bins, kernel_initializer="he_normal", use_bias=False, activation="softmax",
                       name="pred_age")(base_model.output)

    model = Model(inputs=base_model.input, outputs=prediction)

    return model
    
def mean_loss(y_true, y_pred):
    n_bins = NUM_AGE_BINS
    mean_age_true = K.sum(y_true * K.arange(0, n_bins, dtype="float32"), axis=-1)
    mean_age_pred = K.sum(y_pred * K.arange(0, n_bins, dtype="float32"), axis=-1)
    
    mean_loss = K.square(mean_age_pred - mean_age_true)
    
    return K.mean(mean_loss, axis=-1)

def variance_loss(y_true, y_pred):
    n_bins = NUM_AGE_BINS
    mean_age_true = K.sum(y_true * K.arange(0, n_bins, dtype="float32"), axis=-1)
    mean_age_pred = K.sum(y_pred * K.arange(0, n_bins, dtype="float32"), axis=-1)
    
    repeat = K.repeat(K.reshape(mean_age_pred, (-1, 1)), n_bins)
    repeat = K.reshape(repeat, (-1, n_bins))
    
    variance_loss = K.sum(K.square(repeat - K.arange(0, n_bins, dtype="float32"))*y_pred, axis=1)
    
    return K.mean(variance_loss, axis=-1) 

def mean_variance_loss(y_true, y_pred):
    n_bins = NUM_AGE_BINS
    mean_age_true = K.sum(y_true * K.arange(0, n_bins, dtype="float32"), axis=-1)
    mean_age_pred = K.sum(y_pred * K.arange(0, n_bins, dtype="float32"), axis=-1)
    
    repeat = K.repeat(K.reshape(mean_age_pred, (-1, 1)), n_bins)
    repeat = K.reshape(repeat, (-1, n_bins))
    
    mean_loss = K.square(mean_age_pred - mean_age_true)
    
    variance_loss = K.sum(K.square(repeat - K.arange(0, n_bins, dtype="float32"))*y_pred, axis=1)
    
    softmax_loss = K.categorical_crossentropy(y_true, y_pred, from_logits=False)
    
    alpha = MEAN_VARIANCE_ALPHA
    total_loss = alpha*K.mean(mean_loss/n_bins, axis=-1) + (1-alpha)*K.mean(variance_loss, axis=-1) + K.mean(softmax_loss, axis=-1) # scale by 1/n_bins to be in the same scale
    
#     print("ages = {}".format(ages.eval()))
#     print("mean_age_true = {}".format(mean_age_true.eval()))
#     print("mean_age_pred = {}".format(mean_age_pred.eval()))
    
#     print("============================")
#     print("mean_loss = {}".format(mean_loss.eval()))
#     print("variance_loss = {}".format(variance_loss.eval()))
#     print("softmax_loss = {}".format(softmax_loss.eval()))
    
#     print("============================")
#     print("mean_loss_l2norm = {}".format(mean_loss_l2norm.eval()))
#     print("variance_loss_l2norm = {}".format(variance_loss_l2norm.eval()))
#     print("softmax_loss_l2norm = {}".format(softmax_loss_l2norm.eval()))
    
#     print("============================")
#     print("total_loss = {}".format(total_loss.eval()))
    
    return total_loss


def main():
    model = get_model("ResNet50")
    model.summary()


if __name__ == '__main__':
    main()
