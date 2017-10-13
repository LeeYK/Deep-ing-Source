import numpy as np

model_order = 2 # Polynomial order + Intersect

file_1 = np.genfromtxt('data/m4_x.csv', delimiter=',', skip_header=1)
N = np.shape(file_1)[0]
data_x = file_1[:N, 0]
list(map(list, zip(data_x)))
#data_x = np.hstack((np.ones(N).reshape(N, 1), file_1[:N, 0].reshape(N, 1)))
data_y = file_1[:N, 1]
#data_y = np.hstack((np.ones(N).reshape(N, 1), file_1[:N, 1].reshape(N, 1)))

print (data_x)

#data_x = np.linspace(1.0, 10.0, 100)[:, np.newaxis]
#data_y = np.sin(data_x) + 0.1*np.power(data_x,2) + 0.5*np.random.randn(100,1)

#data_x = np.power(data_x, range(model_order))
#data_x /= np.max(data_x, axis=0)

order = np.random.permutation(len(data_x))
portion = 20
test_x = data_x[order[:portion]]
test_y = data_y[order[:portion]]
train_x = data_x[order[portion:]]
train_y = data_y[order[portion:]]

print (test_x)

def get_gradient(w, x, y):
    y_estimate = x.dot(w).flatten()
    error = (y.flatten() - y_estimate)
    mse = (1.0/len(x))*np.sum(np.power(error, 2))
    gradient = -(1.0/len(x)) * error.dot(x)
    return gradient, mse

w = np.random.randn(model_order)
alpha = 0.5
tolerance = 1e-5

# Perform Stochastic Gradient Descent
epochs = 1
decay = 0.95
batch_size = 10
iterations = 0
while True:
    order = np.random.permutation(len(train_x))
    train_x = train_x[order]
    train_y = train_y[order]
    b=0
    while b < len(train_x):
        tx = train_x[b : b+batch_size]
        ty = train_y[b : b+batch_size]
        gradient = get_gradient(w, tx, ty)[0]
        error = get_gradient(w, train_x, train_y)[1]
        w -= alpha * gradient
        iterations += 1
        b += batch_size

    # Keep track of our performance
    if epochs%100==0:
        new_error = get_gradient(w, train_x, train_y)[1]
        print ("Epoch: %d - Error: %.4f" %(epochs, new_error))

        # Stopping Condition
        if abs(new_error - error) < tolerance:
            print ("Converged.")
            break

    epochs += 1
    alpha = alpha * (decay ** int(epochs/1000))
