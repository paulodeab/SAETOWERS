from src.service.signal import Signal
import serial
import time

class LightSignal(Signal):

    def __init__(self):
        self._arduino = serial.Serial('COM10', 9600, timeout=1)
        time.sleep(2)

    def stopSignal(self):
        command: str = '0'
        time.sleep(2)
        self._arduino.write(command.encode())

    def startSignal(self):
        command: str = '1'
        time.sleep(1)
        self._arduino.write(command.encode())
