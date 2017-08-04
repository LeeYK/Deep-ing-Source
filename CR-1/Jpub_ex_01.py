import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Placeholder(X): training set data
# Variable(w): parameter to optimize
# y: result value
# t: training set data

# Modeling
x = tf.placeholder(tf.float32, [None, 5])
w = tf.Variable(tf.zeros([5, 1]))
y = tf.matmul(x, w)
t = tf.placeholder(tf.float32, [None, 1])
loss = tf.reduce_sum(tf.square(y-t))
train_step = tf.train.AdamOptimizer().minimize(loss) # parameter optimization

# Execution(Training)
sess = tf.Session()
sess.run(tf.initialize_all_variables())

train_t = np.array([5.2, 5.7, 8.6, 14.9, 18.2, 20.4, 25.5, 26.4, 22.8, 17.5, 11.1, 6.6])

train_t = train_t.reshape([12,1])

train_x = np.zeros([12, 5])
for row, month in enumerate(range(1, 13)):
	for col, n in enumerate(range(0, 5)):
		train_x[row][col] = month**n

i = 0
for _ in range(100000):
	i += 1
	sess.run(train_step, feed_dict={x:train_x, t:train_t})
	if i % 10000 == 0:
		loss_val = sess.run(loss, feed_dict={x:train_x, t:train_t})
		print ('Step: %d, Loss: %f' % (i, loss_val))

for _ in range(100000):
        i += 1
        sess.run(train_step, feed_dict={x:train_x, t:train_t})
        if i % 10000 == 0:
                loss_val = sess.run(loss, feed_dict={x:train_x, t:train_t})
                print ('Step: %d, Loss: %f' % (i, loss_val))

w_val = sess.run(w)
print w_val

def predict(x):
	result = 0.0
	for n in range(0, 5):
		result += w_val[n][0] * x**n
	return result

fig = plt.figure()
subplot = fig.add_subplot(1,1,1)
subplot.set_xlim(1,12)
subplot.scatter(range(1,13), train_t)
linex = np.linspace(1,12,100)
liney = predict(linex)
subplot.plot(linex,liney)
plt.show()
