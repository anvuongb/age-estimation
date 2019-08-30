import better_exceptions
from keras.applications import ResNet50, InceptionResNetV2, InceptionV3
from keras.layers import Dense, Softmax, Input, Embedding, Lambda
from keras.models import Model
from keras import backend as K
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

def get_model(model_name="ResNet50", n_bins=NUM_AGE_BINS, weights='imagenet', 
              weight_file=None, last_layer_only=False, center_loss=False):
    base_model = None
    if weight_file is not None:
        weights = None

    if weights is not None:
        print("loading weight from {}".format(weights))

    if model_name == "ResNet50":
        base_model = ResNet50(include_top=False, weights=weights, input_shape=(224, 224, 3), pooling="avg")
    elif model_name == "InceptionResNetV2":
        base_model = InceptionResNetV2(include_top=False, weights=weights, input_shape=(299, 299, 3), pooling="avg")
    elif model_name == "InceptionV3":
        base_model = InceptionV3(include_top=False, weights=weights, input_shape=(299, 299, 3), pooling="avg")
    elif model_name == "SEInceptionV3":
        base_model = SEInceptionV3(include_top=False, weights=weights, input_shape=(299, 299, 3), pooling="avg")

    if last_layer_only:
        for layer in base_model.layers:
            layer.trainable=False
    
    if weight_file is None:
        if center_loss is False:
            prediction = Dense(n_bins, kernel_initializer="he_normal", use_bias=False, activation="softmax",
                            name="pred_age")(base_model.output)
            model = Model(inputs=base_model.input, outputs=prediction)
        else:
            fc = Dense(n_bins, kernel_initializer="he_normal", use_bias=False, activation=None,
                            name="fc")(base_model.output)
            prediction = Softmax(name="pred_age")(fc)

            fc_2 = Dense(2, kernel_initializer="he_normal", use_bias=False, activation=None,
                            name="fc_2")(fc)
            input_center = Input(shape=(1,)) # single value ground truth labels as inputs
            centers = Embedding(n_bins,2)(input_center)

            l2_loss = Lambda(lambda x: K.sum(K.square(x[0]-x[1][:,0]),1,keepdims=True),name='l2_loss')([fc_2, centers])
            model = Model(inputs=[base_model.input, input_center], outputs=[prediction, l2_loss])
        return model
    else:
        prediction = Dense(n_bins, kernel_initializer="he_normal", use_bias=False, activation="softmax",
                        name="pred_age")(base_model.output)
        model = Model(inputs=base_model.input, outputs=prediction)
        print("loading weight from {}".format(weight_file))
        model.load_weights(weight_file)

        if center_loss is False:
            return model
        else:
            fc = Dense(n_bins, kernel_initializer="he_normal", use_bias=False, activation=None,
                            name="fc")(model.get_layer(index=-2).output)
            prediction = Softmax(name="pred_age")(fc)

            fc_2 = Dense(2, kernel_initializer="he_normal", use_bias=False, activation=None,
                            name="fc_2")(fc)
            input_center = Input(shape=(1,)) # single value ground truth labels as inputs
            centers = Embedding(n_bins,2)(input_center)

            l2_loss = Lambda(lambda x: K.sum(K.square(x[0]-x[1][:,0]),1,keepdims=True),name='l2_loss')([fc_2, centers])
            model = Model(inputs=[model.input, input_center], outputs=[prediction, l2_loss])
            return model

def get_model_with_gender(model_name="ResNet50", n_bins=NUM_AGE_BINS, weights='imagenet', 
                        weight_file=None, last_layer_only=False, center_loss=False, train_gender_only=True):
    base_model = get_model(model_name=model_name, n_bins=n_bins, weights=weights, 
                           weight_file=weight_file, last_layer_only=last_layer_only, center_loss=center_loss)      

    if train_gender_only:
        for layer in base_model.layers:
            layer.trainable=False

    if center_loss:
        embedding = base_model.get_layer(index=-7).output
        age = base_model.get_layer(name="pred_age").output
    else:
        embedding = base_model.get_layer(index=-2).output
        age = base_model.get_layer(name="pred_age").output

    gender = Dense(1, kernel_initializer="he_normal", 
                use_bias=False, activation="softmax", name="gender")(embedding)
    
    model = Model(inputs=base_model.input, outputs=[gender, age])
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
