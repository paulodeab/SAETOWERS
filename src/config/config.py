MODBUS_CONFIG = {

    "ip" : "192.168.10.86",
    "port" : 502,
    "unit_id" : 1
}


REGISTER = {
    "signal" : 0
}


SOUND = {

    "sound" : "src/sound/danger3.mp3"
}


AI = {

    # Carregar o modelo treinado do YOLO
    "model" : "runs/detect/train28/weights/best.pt"

}