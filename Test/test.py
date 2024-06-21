import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

training_data = np.reshape(np.genfromtxt("Test/train.csv", delimiter=",", skip_header=1),(-1,785))
number_data = training_data[:,0]
image_data = np.divide(np.reshape(np.delete(training_data, 0, 1), (-1,28,28)), 255)
plt.imshow(image_data[3], interpolation='none')
plt.title(str(int(number_data[3])))
plt.show()

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model.fit(image_data, number_data, epochs=10)

probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
# for i in range(10):
prediction = probability_model.predict([image_data[0]])
print(prediction)

# index = int(input("index: "))
# 
# line = np.reshape(np.delete(training_data[index], 0), (-1, 28))
# plt.imshow(line, interpolation='none')
# plt.title(str(int(training_data[index][0])))
# plt.show()

