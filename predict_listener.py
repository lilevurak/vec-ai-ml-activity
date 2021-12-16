from unified_api.subscribe import KafkaListener,PubSubListener
from fn_mnist.predict import FMPredictor

TOPIC = "fn-mnist-predictions"


listener = KafkaListener(topic=TOPIC)

fm_predictor = FMPredictor('fn-mnist-cnn.model')


def callback(msg):
    data = msg['data']
    if "image_path" in data.keys():
        res,res_labels = fm_predictor.predict_image(data['image_path'])
        print("From image path -", res_labels)
    if "imgs" in data.keys():
        imgs_list = data['imgs']
        res,res_labels = fm_predictor.predict_fn_mnist_record(pixels=imgs_list)
        print("From image arrays -", res_labels)


listener.listen(callback)