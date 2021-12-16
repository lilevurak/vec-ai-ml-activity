from abc import (ABC, abstractmethod)
import json
from kafka import KafkaConsumer
from google.cloud import pubsub_v1


class BasicListener(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def listen(self, callback):
        pass


class KafkaListener(BasicListener):
    def __init__(self, topic):
        super(KafkaListener, self).__init__()
        self.consumer = KafkaConsumer(
        topic,
         bootstrap_servers=['localhost:9092'],
         auto_offset_reset='earliest',
         enable_auto_commit=True,
         group_id='fashion-mnist-predictors-cg',
         value_deserializer=lambda x: json.loads(x.decode('utf-8')))

    def listen(self, callback):
        for msg in self.consumer:
            callback(msg.value)


class PubSubListener(BasicListener):
    def __init__(self, topic, project_id):
        super(PubSubListener, self).__init__()
        self.subscriber = pubsub_v1.SubscriberClient()
        self.subscription_path = self.subscriber.subscription_path(project_id, topic)
        self.timeout = 5.0
        self.callback = None

    def msg_callback(self, message):
        data = json.loads(message.data.decode('utf-8'))
        self.callback(data)
        message.ack()

    def listen(self, callback):
        self.callback = callback
        streaming_pull_future = self.subscriber.subscribe(self.subscription_path, callback=self.msg_callback)
        with self.subscriber:
            try:
                streaming_pull_future.result(timeout=self.timeout)
            except TimeoutError:
                streaming_pull_future.cancel()
                streaming_pull_future.result()


