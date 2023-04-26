from abc import ABC, abstractmethod


class IweightFunction(ABC):
    @abstractmethod
    def perform(self):
        pass
