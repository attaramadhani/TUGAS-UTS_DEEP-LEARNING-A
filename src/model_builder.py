"""
Modul Arsitektur Convolutional Neural Network (CNN)
Merancang arsitektur Deep CNN dengan 4 blok konvolusi (Conv2D-ReLU-Pooling),
Flatten, Fully-Connected Dense, dan Dropout dinamis sesuai spesifikasi tugas.
"""

import tensorflow as tf

def build_cnn_model(
    input_shape=(128, 128, 3),
    num_classes=3,
    dropout_rate=0.3,
    pooling_type='max',
    model_name='Custom_Lung_CNN'
):
    """
    Membangun model CNN modular untuk klasifikasi citra CT-Scan paru-paru.
    
    Spesifikasi Arsitektur:
    - Input Layer: (128, 128, 3)
    - Blok 1: Conv2D(32, 3x3) + ReLU + Pooling(2x2)
    - Blok 2: Conv2D(64, 3x3) + ReLU + Pooling(2x2)
    - Blok 3: Conv2D(128, 3x3) + ReLU + Pooling(2x2)
    - Blok 4: Conv2D(128, 3x3) + ReLU + Pooling(2x2)
    - Classifier Head: Flatten + Dense(128, ReLU) + Dropout + Dense(num_classes, Softmax)
    """
    model = tf.keras.Sequential(name=model_name)
    
    # Input Layer
    model.add(tf.keras.layers.Input(shape=input_shape))
    
    # Blok 1
    model.add(tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', name='conv1'))
    if pooling_type.lower() == 'avg':
        model.add(tf.keras.layers.AveragePooling2D((2, 2), name='pool1'))
    else:
        model.add(tf.keras.layers.MaxPooling2D((2, 2), name='pool1'))
        
    # Blok 2
    model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', name='conv2'))
    if pooling_type.lower() == 'avg':
        model.add(tf.keras.layers.AveragePooling2D((2, 2), name='pool2'))
    else:
        model.add(tf.keras.layers.MaxPooling2D((2, 2), name='pool2'))
        
    # Blok 3
    model.add(tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu', name='conv3'))
    if pooling_type.lower() == 'avg':
        model.add(tf.keras.layers.AveragePooling2D((2, 2), name='pool3'))
    else:
        model.add(tf.keras.layers.MaxPooling2D((2, 2), name='pool3'))
        
    # Blok 4
    model.add(tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu', name='conv4'))
    if pooling_type.lower() == 'avg':
        model.add(tf.keras.layers.AveragePooling2D((2, 2), name='pool4'))
    else:
        model.add(tf.keras.layers.MaxPooling2D((2, 2), name='pool4'))
        
    # Classifier Head
    model.add(tf.keras.layers.Flatten(name='flatten'))
    model.add(tf.keras.layers.Dense(128, activation='relu', name='dense_feature'))
    
    if dropout_rate > 0.0:
        model.add(tf.keras.layers.Dropout(dropout_rate, name=f'dropout_{int(dropout_rate*100)}'))
        
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax', name='output_softmax'))
    
    return model

def compile_model(model, optimizer_name='adam', learning_rate=0.0005):
    """
    Melakukan kompilasi model dengan optimizer dan loss function yang ditentukan.
    """
    opt_name = optimizer_name.lower()
    if opt_name == 'adam':
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif opt_name == 'rmsprop':
        opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    elif opt_name == 'sgd':
        opt = tf.keras.optimizers.SGD(learning_rate=learning_rate * 2, momentum=0.9, nesterov=True)
    else:
        raise ValueError(f"Optimizer '{optimizer_name}' tidak didukung. Pilih 'adam', 'rmsprop', atau 'sgd'.")
        
    model.compile(
        optimizer=opt,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
