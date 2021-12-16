import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.python import keras
from keras.utils import np_utils
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPooling2D
import plotly.graph_objs as go
from plotly import tools


IMG_ROWS = 28
IMG_COLS = 28
NUM_CLASSES = 10
TEST_SIZE = 0.2
RANDOM_STATE = 2018
NO_EPOCHS = 50
BATCH_SIZE = 128


def preprocess_data(raw):
    out_y = np_utils.to_categorical(raw.label, NUM_CLASSES)
    num_images = raw.shape[0]
    x_as_array = raw.values[:,1:]
    x_shaped_array = x_as_array.reshape(num_images, IMG_ROWS, IMG_COLS, 1)
    out_x = x_shaped_array / 255
    return out_x, out_y


def create_trace(x, y, ylabel, color):
    trace = go.Scatter(
        x=x, y=y,
        name=ylabel,
        marker=dict(color=color),
        mode="markers+lines",
        text=x
    )
    return trace


def plot_accuracy_and_loss(train_model):
    hist = train_model.history
    acc = hist['accuracy']
    val_acc = hist['val_accuracy']
    loss = hist['loss']
    val_loss = hist['val_loss']
    epochs = list(range(1, len(acc) + 1))

    trace_ta = create_trace(epochs, acc, "Training accuracy", "Green")
    trace_va = create_trace(epochs, val_acc, "Validation accuracy", "Red")
    trace_tl = create_trace(epochs, loss, "Training loss", "Blue")
    trace_vl = create_trace(epochs, val_loss, "Validation loss", "Magenta")

    fig = tools.make_subplots(rows=1, cols=2, subplot_titles=('Training and validation accuracy',
                                                              'Training and validation loss'))
    fig.append_trace(trace_ta, 1, 1)
    fig.append_trace(trace_va, 1, 1)
    fig.append_trace(trace_tl, 1, 2)
    fig.append_trace(trace_vl, 1, 2)
    fig['layout']['xaxis'].update(title='Epoch')
    fig['layout']['xaxis2'].update(title='Epoch')
    fig['layout']['yaxis'].update(title='Accuracy', range=[0, 1])
    fig['layout']['yaxis2'].update(title='Loss', range=[0, 1])
    fig.show()


class Trainer:
    def load_data(self, path):
        train_file = path + "fashion-mnist_train.csv"
        test_file = path + "fashion-mnist_test.csv"
        train_data = pd.read_csv(train_file)
        test_data = pd.read_csv(test_file)

        X, y = preprocess_data(train_data)
        self.X_test, self.y_test = preprocess_data(test_data)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    def train_model_and_validate(self, save_model=True, model_save_path=None):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),
                         activation='relu',
                         kernel_initializer='he_normal',
                         input_shape=(IMG_ROWS, IMG_COLS, 1)))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(64,
                         kernel_size=(3, 3),
                         activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(Dropout(0.4))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(NUM_CLASSES, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer='adam',
                      metrics=['accuracy'])

        train_model = model.fit(self.X_train, self.y_train,
                                batch_size=BATCH_SIZE,
                                epochs=NO_EPOCHS,
                                verbose=1,
                                validation_data=(self.X_val, self.y_val))

        plot_accuracy_and_loss(train_model)
        if save_model:
            model.save(model_save_path)
        self.model = model

    def evaluate_model(self):
        score = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])



if __name__ == "__main__":
    trainer = Trainer()
    trainer.load_data("/home/nidhin/Downloads/fashion-mnist-dataset/")
    trainer.train_model_and_validate(model_save_path="fn-mnist-cnn-test.model")
    trainer.evaluate_model()


