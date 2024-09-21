import argparse
import json
import tensorflow as tf
from tensorflow.keras.models import load_model
from utils import predict

def main():
    parser = argparse.ArgumentParser(description='Predict flower type from an image.')
    parser.add_argument('image_path', type=str, help='Path to the image file')
    parser.add_argument('model_path', type=str, help='Path to the saved Keras model')
    parser.add_argument('--top_k', type=int, default=5, help='Return the top K most likely classes')
    parser.add_argument('--category_names', type=str, help='Path to a JSON file mapping labels to flower names')

    args = parser.parse_args()

    model = load_model(args.model_path, custom_objects={'KerasLayer': tf.keras.layers.Layer})

    probabilities, classes, _ = predict(args.image_path, model, args.top_k)

    if args.category_names:
        with open(args.category_names, 'r') as f:
            class_names = json.load(f)
        class_labels = [class_names[str(cls)] for cls in classes]
    else:
        class_labels = classes

    for i in range(len(probabilities)):
        print(f"{class_labels[i]}: {probabilities[i]:.4f}")

if __name__ == "__main__":
    main()
