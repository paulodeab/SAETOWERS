from src.config.config import SOUND
from src.service.signal import Signal
import pygame

class SoundSignal(Signal):

    def __init__(self):
        pass

    def startSignal(self):
        pygame.mixer.init()
        pygame.mixer.music.load(SOUND.get("sound"))
        pygame.mixer.music.play()

    def stopSignal(self):
        pass


