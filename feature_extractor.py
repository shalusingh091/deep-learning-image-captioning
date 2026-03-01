
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load pretrained ResNet50 model
base_model = ResNet50(weights='imagenet')
model = Model(inputs=base_model.inputs, outputs=base_model.layers[-2].output)

def extract_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    feature = model.predict(img)
    return feature
