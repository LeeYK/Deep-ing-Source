import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# reference: https://www.cs.toronto.edu/~frossard/post/multiple_linear_regression/

#LOAD data

file_1 = np.genfromtxt('data/m4_x.csv', delimiter=',', skip_header=1)
N = np.shape(file_1)[0]
#data_x = file_1[:N, 0]
#list(map(list, zip(data_x)))
data_x_ori = file_1[:N, 0]#np.hstack((np.ones(N).reshape(N, 1), file_1[:N, 0].reshape(N, 1)))
data_y_ori = file_1[:N, 1]

half = round(N)

data_x = data_x_ori[:half].reshape(half, 1)
data_y = data_y_ori[:half].reshape(half, 1)

# NEW DATA
#data_x = np.linspace(1, 8, 100)[:, np.newaxis]
#data_y = np.polyval([1, -14, 59, -70], data_x) \
#        + 1.5 * np.sin(data_x) + np.random.randn(100, 1)
# --------

model_order = 4
data_x = np.power(data_x, range(model_order))
data_x /= np.max(data_x, axis=0)

#data_x_ori = np.power(data_x_ori, range(model_order))
data_x_ori /= np.max(data_x_ori, axis=0)

order = np.random.permutation(len(data_x))
portion = round(half/2)
test_x = data_x[order[:portion]]
test_y = data_y[order[:portion]]
train_x = data_x[order[portion:]]
train_y = data_y[order[portion:]]

with tf.name_scope("IO"):
    inputs = tf.placeholder(tf.float32, [None, model_order], name="X")
    outputs = tf.placeholder(tf.float32, [None, 1], name="Yhat")

with tf.name_scope("LR"):
    W = tf.Variable(tf.zeros([model_order, 1], dtype=tf.float32), name="W")
    y = tf.matmul(inputs, W)

with tf.name_scope("train"):
    learning_rate = tf.Variable(0.5, trainable=False)
    cost_op = tf.reduce_mean(tf.pow(y-outputs, 2))
    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_op)

    tolerance = 1e-4

# Perform Stochastic Gradient Descent
epochs = 1
last_cost = 0
alpha = 0.3
max_epochs = 50000

sess = tf.Session() # Create TensorFlow session
print ("Beginning Training")
with sess.as_default():
    #init = tf.initialize_all_variables()
    init = tf.global_variables_initializer()
    sess.run(init)
    sess.run(tf.assign(learning_rate, alpha))
    while True:
        # Execute Gradient Descent
        sess.run(train_op, feed_dict={inputs: train_x, outputs: train_y})

        # Keep track of our performance
        if epochs%100==0:
            cost = sess.run(cost_op, feed_dict={inputs: train_x, outputs: train_y})
            print ("Epoch: %d - Error: %.4f" %(epochs, cost))

            # Stopping Condition
            if abs(last_cost - cost) < tolerance or epochs > max_epochs:
                print ("Converged.")
                break
            last_cost = cost

        epochs += 1

    w = W.eval()
    print ("w =", w)
    print ("Test Cost =", sess.run(cost_op, feed_dict={inputs: test_x, outputs: test_y}))


tt = np.linspace(np.min(0), np.max(1))
#plt.plot(tt, w[4]*tt**4+w[3]*tt**3+w[2]*tt**2+w[1]*tt+w[0])
plt.plot(tt, w[3]*tt**3+w[2]*tt**2+w[1]*tt+w[0])
#plt.plot(tt, w[2]*tt**2+w[1]*tt+w[0])
plt.plot(data_x_ori, data_y_ori, 'kx')

#tt = np.linspace(np.min(0), np.max(2))
#bf_line = w[0]*tt


plt.xlabel('frame')
plt.ylabel('X')
#plt.title('Regression')

plt.savefig('mpg.png')

plt.show()
