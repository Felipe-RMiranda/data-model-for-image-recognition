import tensorflow as tf

def convertandsave(model):
    # Converter o modelo para TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    # Salvar o modelo TFLite
    with open('meu_modelo.tflite', 'wb') as f:
        f.write(tflite_model)

class ModelExported:

    def __init__(self, model):
        self.model = model
        convertandsave(self.model)