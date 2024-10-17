from itertools import count

from pymodbus.client import ModbusTcpClient

from main2 import result
from src.config.config import MODBUS_CONFIG
from src.service.signal import Signal


class SignalModbus(Signal):

    _alert: bool
    _client: ModbusTcpClient

    def __init__(self):
        super().__init__()
        self._alert = False
        self._client = ModbusTcpClient(MODBUS_CONFIG.get("ip"), MODBUS_CONFIG.get("port"))
        self._unit_id = MODBUS_CONFIG.get("")


    def connect(self) -> bool:
        return self._client.connect()

    def disconnect(self):
        self._client.close()

    def startSignal(self):
        self._alert = True
        print("Sinal Ativado")

        self._client.write_register(address=0, value=1, slave=self._unit_id)

    def stopSignal(self):
        self._alert = False
        self._client.write_register(address=0, value=0, slave=self._unit_id)

    def readSignal(self) -> bool:
        response = self._client.read_holding_registers(address=0, count=1, slave=self._unit_id)
        if not response.isError():
            self._alert = bool(response.registers[0])
            return self._alert
        else:
            print("Erro ao ler registro Modbus")
            return False




