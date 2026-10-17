import tensorflow as tf

tf.random.set_seed(42) 

trainData = tf.keras.utils.image_dataset_from_directory( "dataset/train",image_size=(128, 128), batch_size=32)

valData = tf.keras.utils.image_dataset_from_directory("dataset/val", image_size=(128, 128), batch_size=32, shuffle=False)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(128, 128, 3)),
    tf.keras.layers.Rescaling(1.0 / 255),

    tf.keras.layers.Conv2D(32, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(5, activation="softmax")
])

model.compile(optimizer="sgd", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(trainData, validation_data=valData, epochs=10)

model.save("object_model.keras")
print("Training completed!")
