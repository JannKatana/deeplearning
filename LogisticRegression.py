import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage


def load_dataset():
	train_dataset = h5py.File('datasets/train_catvnoncat.h5', 'r')
	train_set_x_orig = np.array(train_dataset["train_set_x"][:])
	train_set_y_orig = np.array(train_dataset["train_set_y"][:])

	test_dataset = h5py.File('datasets/test_catvnoncat.h5', 'r')
	test_set_x_orig = np.array(test_dataset["test_set_x"][:])
	test_set_y_orig = np.array(test_dataset["test_set_y"][:])

	classes = np.array(test_dataset["list_classes"][:])

	train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
	test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

	return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def show_image(image):
	plt.imshow(image)
	plt.show()


def sigmoid(z):
	s = 1/(1+np.exp(-1*z))
	return s


def initialize_with_zeros(dim):
	w = np.zeros((dim, 1), dtype=float)
	b = 0
	return w, b


def propagate(w, b, X, Y):
	m = X.shape[1]

	# Forward propagation
	Z = np.dot(w.T, X) + b
	A = sigmoid(Z)
	cost = (-1/m)*(np.sum(Y*np.log(A) + (1-Y)*np.log(1-A)))

	# Back propagation
	dw = (1/m)*np.dot(X, (A-Y).T)
	db = (1/m)*np.sum(A-Y)

	cost = np.squeeze(cost)

	grads = {
		"dw": dw,
		"db": db
	}

	return grads, cost


def optimize(w, b , X, Y, iterations, learning_rate, print_cost = False):
	costs = []

	for i in range(iterations):
		# Calculate grads and cost
		grads, cost = propagate(w, b, X, Y)

		# Retrieve derivatives from grads
		dw = grads["dw"]
		db = grads["db"]

		# Update gradients
		w = w - (learning_rate*dw)
		b = b - (learning_rate*db)

		# Record the cost every 100 iterations
		if i % 100 == 0:
			costs.append(cost)

		# Print cost every 100 100 training iterations
		if print_cost and i % 100 == 0:
			print("Cost after iteration %i : %f" % (i, cost))

	params = {
		"w": w,
		"b": b
	}

	grads = {
		"dw": dw,
		"db": db
	}

	return params, grads, costs


def predict(w, b, X):
	m = X.shape[1]
	Y_prediction = np.zeros((1, m))
	w = w.reshape(X.shape[0], 1)

	A = sigmoid(np.dot(w.T, X) + b)

	for i in range(A.shape[1]):
		if A[0, i] <= 0.5:
			Y_prediction[0, i] = 0
		else:
			Y_prediction[0, i] = 1

	return Y_prediction


def model(X_train, Y_train, X_test, Y_test, iterations = 2000, learning_rate = 0.5, print_cost = False):
	# Initialize parameters with zeros
	w, b = initialize_with_zeros(X_train.shape[0])

	# Gradient descend
	parameters, grads, costs = optimize(w, b, X_train, Y_train, iterations, learning_rate, print_cost)

	w = parameters["w"]
	b = parameters["b"]

	# Predict test/train set examples
	Y_prediction_train = predict(w, b, X_train)
	Y_prediction_test = predict(w, b, X_test)

	# Print training and test errors
	print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
	print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

	d = {
		"costs": costs,
		"Y_prediction_train": Y_prediction_train,
		"Y_prediction_test": Y_prediction_test,
		"w": w,
		"b": b,
		"learning_rate": learning_rate,
		"iterations": iterations
	}

	return d


def plot_learning_curve(d):
	costs = np.squeeze(d["costs"])
	plt.plot(costs)
	plt.ylabel("costs")
	plt.xlabel("iterations (per hundreds)")
	plt.title("learning rate = %f" % d["learning_rate"])
	plt.show()

if __name__ == "__main__":
	# Load data
	train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

	# Get number of samples and image dimensions
	m_train = train_set_x_orig.shape[0]
	m_test = test_set_x_orig.shape[0]
	image_dim = train_set_x_orig.shape[1]

	# Flatten the training and test samples
	train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
	test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

	# Standardize dataset
	train_set_x = train_set_x_flatten/255
	test_set_x = test_set_x_flatten/255

	d = model(train_set_x, train_set_y, test_set_x, test_set_y, iterations=2000, learning_rate=0.005, print_cost=True)

	# Plot learning curve
	plot_learning_curve(d)


