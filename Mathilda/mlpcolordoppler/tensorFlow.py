import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from keras.constraints import maxnorm

no_of_layers = 100
callbacks = [
  # Interrupt training if `val_loss` stops improving for over 2 epochs
  tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
  # Write TensorBoard logs to `./logs` directory
  tf.keras.callbacks.TensorBoard(log_dir='./logs')
]

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dropout(0.05, input_shape=(500,)))
for layer in range(no_of_layers):
    model.add(layers.Dense(100, activation='tanh', kernel_constraint=maxnorm(3)))
model.add(layers.Dense(500))

model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
              loss=tf.keras.losses.mean_squared_error,
              metrics=['accuracy'])

inputs = np.load("inputs_good_medium.npy")
targets = np.load("targets_good_medium.npy")
x = np.load("x.npy")

training_inputs, testing_inputs, training_targets, testing_targets = train_test_split(inputs, targets, test_size=0.3)

model.fit(training_inputs, training_targets, epochs=10, batch_size=5, callbacks=callbacks)

prediction = model.predict(testing_inputs)

for index in range(len(prediction)):
    plt.plot(x, prediction[index])
    plt.plot(x, testing_targets[index])
    plt.show()
