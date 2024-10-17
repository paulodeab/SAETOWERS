from abc import ABC, abstractmethod

class Signal(ABC):

    _alert: bool

    @abstractmethod
    def startSignal(self):
        pass

    @abstractmethod
    def stopSignal(self):
        pass





    
