import tensorflow as tf

def build_model():
    batch_size = 16
    img_height = 50
    img_width  = 50

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=16,kernel_size=(3,3),padding="same",activation="relu",input_shape=[img_width, img_height, 3]),
        tf.keras.layers.Conv2D(filters=16,kernel_size=(3,3),padding="same",activation="relu"),

        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Conv2D(filters=16,kernel_size=(3,3),padding="same",activation="relu"),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(filters=16,kernel_size=(3,3),padding="same",activation="relu"),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(2, activation="softmax")
    ])

    # 编译
    model.summary()

    model.compile(optimizer="adam",
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

    return model

build_model()