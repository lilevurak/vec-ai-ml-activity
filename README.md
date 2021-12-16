# ML Activity

## Fashion MNIST CNN Classifier

### Training

For training the classifier. 
1. Download the Fashion Mnist Dataset
2. Run **python fn_mnist/trainer.py** 

For training on a different dataset you can tweak the **load_data"** function found in the class **Trainer**. You can load directly from another csv like done currently or read images from a folder using opencv and convert it to ndarrays and load a dataframe. Update the **IMG_ROWS, IMG_COLS, NUM_CLASSES** values according to the data.


### Prediction

For prediction you can import the **FMPredictor** class from fn_mnist/predict.py. You can pass the model path to its constructer to load the model. In it there are multiple versions of predict functions for different input types, it can take ndarrays of dimenions nx28x28x1 or 2d lists of size nx784 where n is the number of images.It can also accept a single image's file path and load it using opencv.


## Unified API for Kafka/PubSub

Under **unified_api** you'll find publisher and listener implementions for both Kafka and PubSub. You can use the publisher to send/stream data to topics in a non blocking manner. And with the listener we can listen for new data/events on these topics. All publishers and listeners should inherit the abstract classes BasicPublisher and BasicListener respectively. You can pass a json serializable dict as the message payload.

## Async ML Predictions Application.

Either Kafka or PubSub can be used as the underlying message broker for the prediction service workflow. In **data_sender.py** a publisher is used to push prediction requests asynchronously into a topic. Both file path based and 2d array based requests done as examples. In **predict_listener.py** you can see a listener being used for loading the CNN model and process the requests.

After setting up Kafka/PubSub, run **python predict_listener.py** and **python data_sender.py** in different terminals.

The prediction listeners can be scaled up by running on multiple instances since they subscribe to the same topic. But additional configuration must be done at the Kafka/PubSub service level to optimize it for the required scenarios. Currently the prediction listener is only printing the results. But these can be easily streamed to another topic and can be consumed by database listeners which make the required db read/writes asynchronously.