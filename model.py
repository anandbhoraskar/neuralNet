import numpy as np
import random

class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta):
        n = len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            print "Epoch {0} complete".format(j)

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] 
        zs = [] # list to store all the z vectors
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        test_results = [np.argmax(self.feedforward(x))
                        for x in test_data]
        return test_results


    def cost_derivative(self, output_activations, y):
        return (output_activations-y)

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

def as_vector(n):
    vec = np.zeros((10,1))
    vec[n] = 1
    return vec

def vectorize_inp(r):
    vec = np.zeros((85,1))
    p = 0
    it = iter(r)
    for x in it:
        vec[p+x-1] = 1
        vec[p+4+next(it)-1] = 1
        p += 17
    return vec

def print_counts(r):
    print [r.count(i) for i in range(10)]

if __name__ == "__main__":
    train_data = np.loadtxt("train.csv", delimiter=',', skiprows=1)
    train_input = [(np.reshape(vectorize_inp(row[:-1]), (85,1)), as_vector(row[10])) for row in train_data]
    net = Network([85, 18, 10])
    net.SGD(train_input, epochs=10, mini_batch_size=10, eta=3)
    test_data = np.loadtxt("test1.csv", delimiter=',', skiprows=1)
    test_input = [np.reshape(vectorize_inp(row[1:]), (85,1)) for row in test_data]
    indices = [row[0] for row in test_data]
    result = net.evaluate(test_input)
    output_array = zip(indices, result)
    np.savetxt(fname="result.csv",fmt="%d", X=output_array,
               delimiter=',', header="id,CLASS", comments="")



