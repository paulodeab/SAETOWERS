from src.signal import Signal


class SignalModbus(Signal):

    _alert: bool

    def __init__(self):
        self._alert = False


    def startSignal(self):
        self._alert = True




