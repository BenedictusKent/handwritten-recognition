import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# open training file and put in list
train_file = open("src/mnist_train_100.csv", 'r')
train_read = train_file.readlines()
train_file.close()

# open test file and put in list
test_file = open("src/mnist_test_10.csv", 'r')
test_read = test_file.readlines()
test_file.close()

# separate pixel value and label of train images
val = []
lab = []
for i in range(len(train_read)):
    val.append([])
    val[i] = train_read[i].split(',')
    lab.append([])
    lab[i] = val[i][0]
    val[i].pop(0)

# change pixel value and label to numpy of train images
x_train = np.array(val).astype(float)
x_train.shape = (100, 28, 28)
y_train = np.array(lab).astype(float)

# separate pixel value and label of test images
val.clear()
lab.clear()
for i in range(len(test_read)):
    val.append([])
    val[i] = test_read[i].split(',')
    lab.append([])
    lab[i] = val[i][0]
    val[i].pop(0)

# change pixel value and label to numpy of test images
x_test = np.array(val).astype(float)
x_test.shape = (10, 28, 28)
y_test = np.array(lab).astype(float)

# normalize train and test images
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# visualization
plt.imshow(x_train[0], cmap='Greys')
plt.show()

# model
model = tf.keras.Sequential(
    [
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ]
)

# compile
model.compile(
    optimizer=tf.keras.optimizers.Adam(lr=0.01),
    loss='categorical_crosssentropy',
    metrics=['accuracy']
)

model.fit(x_train, y_train, epochs=3)