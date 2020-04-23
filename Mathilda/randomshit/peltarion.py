import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Save to h5 file in peltarion and load using keras
model = tf.keras.models.load_model("experiment_2.h5", compile=False)

inputs = np.load("../mlpcolordoppler/inputs_good_medium_pelt.npy")
targets = np.load("../mlpcolordoppler/targets_good_medium_pelt.npy")

model.compile(optimizer='sgd', loss='mean_squared_error')

print(model.summary())
print(model.evaluate(inputs, targets))

prediction = model.predict(inputs)

plt.plot(prediction[0][:, 0], prediction[0][:, 1])
plt.show()
