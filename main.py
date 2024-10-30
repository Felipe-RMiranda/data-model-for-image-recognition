import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory
import ModelExported

dataset = image_dataset_from_directory(
    'modelo-dados-waterpollution',
    image_size=(224, 224),
    batch_size=32,
    label_mode='categorical'
)

# Definir um modelo básico (MobileNetV2)
base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = True

model = models.Sequential([  # Cria um modelo sequencial, onde as camadas são empilhadas uma após a outra.
    base_model, #Adiciona o modelo pré-treinado (neste caso, MobileNetV2) como a primeira camada. Ele extrai características das imagens.
    layers.GlobalAveragePooling2D(), # Esta camada faz uma média global das características extraídas
    layers.Dense(128, activation='relu'), #Adiciona uma camada densa (totalmente conectada) com 128 neurônios e a função de ativação ReLU
    layers.Dense(1, activation='softmax')  #camada com 2 neurônios (para 2 classes de lixo) e a função de ativação softmax. Ela gera probabilidades para cada classe
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) # Compilar o modelo

model.fit(dataset, epochs=2) # Treinar o modelo do dataset fornecido com a quantidade de epocas que modelo sera treinado

# Salvar o modelo em formato TensorFlow SavedModel
#model.save('model')

#exported_model = ModelExported(model)