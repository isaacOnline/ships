from abc import ABC, abstractmethod


class ModelRunner(ABC):
    @abstractmethod
    def __init__(self):
        pass

    def fit(self, *pos_args, **named_args):
        return self.model.fit(*pos_args, **named_args)

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def save(self):
        pass

    def compile(self):
        self.model.compile(optimizer=self.optimizer, loss=self.loss)
