import numpy as np
from PIL import Image
import tensorflow as tf

def process_image(image):
    image = tf.convert_to_tensor(image)
    image = tf.image.resize(image, [224, 224])
    image = image / 255.0
    return image.numpy()

def predict(image_path, model, top_k=5):
    with Image.open(image_path) as img:
        img = np.asarray(img)

    processed_image = process_image(img)
    image_batch = np.expand_dims(processed_image, axis=0)
    prediction = model.predict(image_batch)

    probabilities, classes = tf.nn.top_k(prediction, k=top_k)
    probs_list = list(probabilities.numpy()[0])
    classes_list = list(classes.numpy()[0])

    return probs_list, classes_list, processed_image
