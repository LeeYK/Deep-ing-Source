import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

num_points = 1000
x_data = []
y_data = []
vectors_set = []

F1 = open("data_x.txt", "r")
F2 = open("data_x_nn.txt", "w")

data = F1.readlines()

for line in data:
    words = line.split()
    x1 = float(words[0])
    y1 = float(words[1])
    vectors_set.append([x1, y1])
    #x_data.append(float(words[0]))
    #y_data.append(float(words[1]))
    #vectors_set.append([float(words[0]), float(words[1])])

    x_data = [v[0] for v in vectors_set]
    y_data = [v[1] for v in vectors_set]

#for i in range(num_points):
#    x1 = np.random.normal(0.0, 0.55)
#    y1 = x1 * 0.1 + 0.3 + np.random.normal(0.0, 0.03)
#    vectors_set.append([x1, y1])

#x_data = [v[0] for v in vectors_set]
#y_data = [v[1] for v in vectors_set]

#graphical expression
plt.plot(x_data, y_data, 'ro')
plt.title('data')
plt.legend()
plt.show()

#X = tf.placeholder(tf.float32, shape=[None, 1])
#Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_uniform([1], -5.0, 5.0), name='weight')
b = tf.Variable(tf.random_uniform([1], -25.0, 25.0), name='bias')
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b
#hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

loss = tf.reduce_mean(tf.square(y - y_data))
#loss = tf.reduce_mean(tf.square(tf.squeeze(y) - y_data))
#cost = -tf.reduce_mean(y_data * tf.log(y) + (1 - y_data) * (tf.log(1 - y)))
#loss = tf.reduce_mean(-tf.reduce_sum(y_data * tf.log(y) + (1.0-y_data) * tf.log(1.0-y) ))
#loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_data) + (1-y) * tf.log(1-y_data) ))
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for step in range(2001):
    sess.run(train)
    if step % 200 == 0:
        print(step, sess.run(W), sess.run(b))
        print(step, sess.run(loss))

        #graphical expression
        plt.plot(x_data, y_data, 'ro')
        plt.plot(x_data, sess.run(W) * x_data + sess.run(b))
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.show()
