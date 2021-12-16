import pandas as pd
from unified_api.publish import KafkaPublisher,PubSubPublisher

TOPIC = "fn-mnist-predictions"

publisher = KafkaPublisher()

PATH = "/home/nidhin/Downloads/fashion-mnist-dataset/"
test_file = PATH + "fashion-mnist_test.csv"
test_data = pd.read_csv(test_file)

test_pixels = test_data.values[:, 1:]
imgs = []
i = 1
publisher.publish(TOPIC, {"image_path":"test-images/pullover-1.jpg"})


for tp in test_pixels:
    plist = tp.tolist()
    imgs.append(plist)
    i += 1
    if i%10 == 0:
        data = {"imgs": imgs}
        publisher.publish(TOPIC, data)
        imgs = []

publisher.publish(TOPIC, {"imgs": imgs})
publisher.close()



