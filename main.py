import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory

# Carregar o conjunto de dados de imagens
dataset = image_dataset_from_directory(
    '/home/anonymous/Imagens/modelo-dados/waterpollution',
    image_size=(224, 224),
    batch_size=32,
    label_mode='categorical'
)

# Definir um modelo básico (MobileNetV2)
base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dense(2, activation='softmax')  # 5 classes de lixo
])

# Compilar o modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Treinar o modelo
model.fit(dataset, epochs=10)

