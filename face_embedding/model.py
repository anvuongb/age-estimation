from keras.engine import  Model
from keras.layers import Input
from keras_vggface.vggface import VGGFace

def get_model(model_name="ResNet50", feature_layer=None):
    base_model = None

    if model_name == "ResNet50":
        base_model = VGGFace(model="resnet50", include_top=False, input_shape=(224, 224, 3), pooling='avg')
    if model_name == "VGG16":
        base_model = VGGFace(model="vgg16", include_top=False, input_shape=(224, 224, 3), pooling='avg')
    if model_name == "SENet50":
        base_model = VGGFace(model="senet50", include_top=False, input_shape=(224, 224, 3), pooling='avg')

    if feature_layer is not None:
        out = base_model.get_layer(feature_layer).output
        base_model = Model(base_model.input, out)

    return base_model

if __name__ == "__main__":
    model = get_model()
    model.summary()