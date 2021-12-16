from tensorflow.python import keras
from keras.utils import np_utils
import numpy as np
import pandas as pd
import cv2
IMG_ROWS = 28
IMG_COLS = 28
NUM_CLASSES = 10


class FMPredictor():
    def __init__(self, model_path):
        self.model = keras.models.load_model(model_path)
        self.labels = {0 : "T-shirt/top", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat",
          5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle Boot"}

    def predict(self, imgs):
        res = np.argmax(self.model.predict(imgs), axis=-1)
        labels = [self.labels[x] for x in res]
        return res, labels

    def predict_fn_mnist_record(self, pixels):
        raw = np.asarray(pixels)
        if raw.ndim == 1:
            raw = np.expand_dims(raw, axis=0)
        num_images = raw.shape[0]
        x_shaped_array = raw.reshape(num_images, IMG_ROWS, IMG_COLS, 1)
        out_x = x_shaped_array / 255
        return self.predict(out_x)

    def predict_image(self, img_path):
        img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
        img_resized = cv2.resize(img, (IMG_ROWS,IMG_COLS),cv2.INTER_AREA)
        img_resized = img_resized / 255
        x = np.expand_dims(np.expand_dims(img_resized, axis=0), axis=3)
        return self.predict(x)


if __name__ == "__main__":
    predictor = FMPredictor("../fn-mnist-cnn.model")

    res, res_labels = predictor.predict_image("../test-images/pullover-1.jpg")
    print(res_labels)

    PATH = "/home/nidhin/Downloads/fashion-mnist-dataset/"
    test_file = PATH + "fashion-mnist_test.csv"
    test_data = pd.read_csv(test_file)
    test_pixels = test_data.values[:,1:]
    imgs = []
    i = 0
    for tp in test_pixels:
        plist = tp.tolist()
        imgs.append(plist)
        i+=1
        if i ==10:
            break

    ans,ans_labels = predictor.predict_fn_mnist_record(imgs)
    print(ans_labels)




