from abc import (ABC, abstractmethod)
import json
from kafka import KafkaProducer
from google.cloud import pubsub_v1


class BasicPublisher(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def publish(self, topic, message):
        pass

    @abstractmethod
    def close(self):
        pass


class KafkaPublisher(BasicPublisher):
    def __init__(self):
        super().__init__()
        self.producer = KafkaProducer(bootstrap_servers=['localhost:9092'],value_serializer=lambda x: json.dumps(x).encode('utf-8'))

    def publish(self, topic, message):
        data = {'data': message}
        self.producer.send(topic, value=data)

    def close(self):
        self.producer.flush()
        self.producer.close()


class PubSubPublisher(BasicPublisher):
    def __init__(self, project_id):
        super().__init__()
        self.publisher = pubsub_v1.PublisherClient()
        self.project_id = project_id

    def publish(self, topic, message):
        topic_path = self.publisher.topic_path(self.project_id, topic)
        data = {'data': message}
        self.publisher.publish(topic_path,json.dumps(data).encode("utf-8"))

    def close(self):
        pass
