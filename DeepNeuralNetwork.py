__author__ = 'tan_nguyen'
import numpy as np
import numpy.matlib
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
import pickle

def generate_data():
    '''
    generate data
    :return: X: input data, y: given labels
    '''
    np.random.seed(0)
    X, y = datasets.make_circles(200, noise=0.010)
    return X, y


def plot_decision_boundary(pred_func, X, y):
    '''
    plot the decision boundary
    :param pred_func: function used to predict the label
    :param X: input data
    :param y: given labels
    :return:
    '''
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.show()


########################################################################################################################
########################################################################################################################
# YOUR ASSSIGMENT STARTS HERE
# FOLLOW THE INSTRUCTION BELOW TO BUILD AND TRAIN A 3-LAYER NEURAL NETWORK
########################################################################################################################
########################################################################################################################
class NeuralNetwork(object):
    """
    This class builds and trains a neural network
    """

    def __init__(self, nn_input_dim, nn_hidden_dim, nn_output_dim, nn_num_of_layers, actFun_type='tanh', reg_lambda=0.01, seed=0):
        '''
        :param nn_input_dim: input dimension
        :param nn_hidden_dim: the number of hidden units
        :param nn_output_dim: output dimension
        :param actFun_type: type of activation function. 3 options: 'tanh', 'sigmoid', 'relu'
        :param reg_lambda: regularization coefficient
        :param seed: random seed
        '''

        if nn_num_of_layers <= 2:
            raise Exception('You cannot have a Neural Network with no Hidden Layers!!')

        self.nn_input_dim = nn_input_dim
        self.nn_hidden_dim = nn_hidden_dim
        self.nn_output_dim = nn_output_dim
        self.actFun_type = actFun_type
        self.reg_lambda = reg_lambda
        self.nn_num_of_layers = nn_num_of_layers - 1
        # initialize the weights and biases in the network
        np.random.seed(seed)

        self.Layer_array = []
        for i in range(0,self.nn_num_of_layers):
            if i==(self.nn_num_of_layers-1):
                x = Layer(nn_hidden_dim, nn_output_dim, 4560*i, 'last')
                self.Layer_array.append(x)
            elif i == 0:
                x = Layer(nn_input_dim, nn_hidden_dim, 4560*i, 'middle')
                self.Layer_array.append(x)
            else:
                x = Layer(nn_hidden_dim, nn_hidden_dim,  4560*i, 'middle')
                self.Layer_array.append(x)


    def actFun(self, z, type):
        '''
        actFun computes the activation functions
        :param z: net input
        :param type: Tanh, Sigmoid, or ReLU
        :return: activations
        '''

        # YOU IMPLEMENT YOUR actFun HERE
        if type == 'tanh':
            val = (np.exp(2*z)-1)/(np.exp(2*z)+1)
        elif type == 'sigmoid':
            val = 1/(1+np.exp(-1*z))
        elif type == 'relu':
            val =  z*(z>0)
        else :
            raise Exception('Oh My God you have an activation function that does not exist :-/ !!')
        return val

    def diff_actFun(self, z, type):
        '''
        diff_actFun computes the derivatives of the activation functions wrt the net input
        :param z: net input
        :param type: Tanh, Sigmoid, or ReLU
        :return: the derivatives of the activation functions wrt the net input
        '''

        # YOU IMPLEMENT YOUR diff_actFun HERE
        if type == 'tanh':
            value = np.tanh(z)
            diff_val = 1-value*value
            #diff_val = -1* ( (np.exp(2 * z) - 1) / (np.exp(2 * z) + 1)**2 ) * (np.exp(2*z) * 2) + 2*(np.exp(2*z)-1)/(np.exp(2*z)+1)
        elif type == 'sigmoid':
            diff_val = (1 / (1 + np.exp(-1 * z))**2) *  np.exp(-1 * z)
        elif type == 'relu':
            diff_val = (z>=0)*1
        else:
            raise Exception('Oh My God you have an activation function that does not exist :-/ !!')
        return diff_val

    def feedforward(self, X):
        '''
        feedforward builds a 3-layer neural network and computes the two probabilities,
        one for class 0 and one for class 1
        :param X: input data
        :param actFun: activation function
        :return:
        '''

        # YOU IMPLEMENT YOUR feedforward HERE
        for i in range(0,self.nn_num_of_layers):
            if i==0 :
                self.Layer_array[0].feedforward(X, lambda x: self.actFun(x, type=self.actFun_type))
            else:
                self.Layer_array[i].feedforward(self.Layer_array[i - 1].a, lambda x: self.actFun(x, type=self.actFun_type))

        self.probs = self.Layer_array[self.nn_num_of_layers - 1].probs
        return None

    def calculate_loss(self, X, y):
        '''
        calculate_loss computes the loss for prediction
        :param X: input data
        :param y: given labels
        :return: the loss for prediction
        '''
        num_examples = len(X)
        self.feedforward(X)
        # Calculating the loss

        # YOU IMPLEMENT YOUR CALCULATION OF THE LOSS HERE
        data_loss = 0
        for i in range(num_examples):
            for j in range(self.nn_output_dim):
                if y[i] == j:
                    data_loss  += np.log(self.probs[i][j])

        data_loss = -1 * data_loss

        # Add regulatization term to loss (optional)

        temp = 0
        for i in range(0,self.nn_num_of_layers):
            temp += np.sum(np.square(self.Layer_array[i].W))
        data_loss += self.reg_lambda / 2 * temp
        return (1. / num_examples) * data_loss

    def predict(self, X):
        '''
        predict infers the label of a given data point X
        :param X: input data
        :return: label inferred
        '''
        self.feedforward(X)
        return np.argmax(self.probs, axis=1)

    def backprop(self, X, y):
        '''
        backprop implements backpropagation to compute the gradients used to update the parameters in the backward step
        :param X: input data
        :param y: given labels
        :return: dL/dW1, dL/b1, dL/dW2, dL/db2
        '''

        dW = []
        db = []
        for i in range(0,self.nn_num_of_layers):
            temp_mat = np.zeros(self.Layer_array[i].W.shape)
            dW.append(temp_mat)
            temp_vec = np.zeros(self.Layer_array[i].b.shape)
            db.append(temp_vec)

        # IMPLEMENT YOUR BACKPROP HERE
        num_examples = len(X)

        #dW for last layer
        for i in range(self.nn_num_of_layers - 1 , -1 , -1):
            if i == (self.nn_num_of_layers - 1):
                delta3 = self.probs
                delta3[range(num_examples), y] -= 1
                dW[i] = (self.Layer_array[i-1].a.T).dot(delta3)
                db[i] = np.sum(delta3, axis=0, keepdims=True)

            elif i == 0:
                #dW for previous layers
                delta2 = delta3.dot(self.Layer_array[i+1].W.T) * self.diff_actFun(self.Layer_array[i].z, self.actFun_type)
                dW[i] = np.dot(X.T, delta2)
                db[i] = np.sum(delta2, axis=0)

            else:

                # dW for previous layers
                delta2 = delta3.dot(self.Layer_array[i + 1].W.T) * self.diff_actFun(self.Layer_array[i].z,self.actFun_type)
                dW[i] = np.dot(self.Layer_array[i-1].a.T, delta2)
                db[i] = np.sum(delta2, axis=0)
                delta3 = delta2

        return dW, db

    def fit_model(self, X, y, epsilon=0.001, num_passes=20000, print_loss=True):
        '''
        fit_model uses backpropagation to train the network
        :param X: input data
        :param y: given labels
        :param num_passes: the number of times that the algorithm runs through the whole dataset
        :param print_loss: print the loss or not
        :return:
        '''
        # Gradient descent.
        for t in range(0, num_passes):
            # Forward propagation
            self.feedforward(X)
            # Backpropagation
            dW, db = self.backprop(X, y)

            for i in range(0,self.nn_num_of_layers):
                # Add regularization terms (b1 and b2 don't have regularization terms)
                dW[i] += self.reg_lambda * self.Layer_array[i].W

                # Gradient descent parameter update
                self.Layer_array[i].W += -epsilon * dW[i]
                self.Layer_array[i].b += -epsilon * db[i]


            # Optionally print the loss.
            # This is expensive because it uses the whole dataset, so we don't want to do it too often.
            if print_loss and t % 1000 == 0:
                print("Loss after iteration %i: %f" % (t, self.calculate_loss(X, y)))

    def visualize_decision_boundary(self, X, y):
        '''
        visualize_decision_boundary plots the decision boundary created by the trained network
        :param X: input data
        :param y: given labels
        :return:
        '''
        plot_decision_boundary(lambda x: self.predict(x), X, y)


class Layer(object):
    def __init__(self ,nn_input_dim, nn_hidden_dim,  seed=0, layer_Type='middle'):
        self.nn_input_dim = nn_input_dim
        self.nn_hidden_dim = nn_hidden_dim

        self.dW = 0
        self.db = 0

        self.layer_Type = layer_Type


        # initialize the weights and biases in the network
        np.random.seed(seed)
        self.W = np.random.randn(self.nn_input_dim, self.nn_hidden_dim) / np.sqrt(self.nn_input_dim)
        self.b = np.zeros((1, self.nn_hidden_dim))



    def feedforward(self, X, actFun):
        '''
        feedforward builds a 3-layer neural network and computes the two probabilities,
        one for class 0 and one for class 1
        :param X: input data
        :param actFun: activation function
        :return:
        '''

        self.z = np.dot(X, self.W) + self.b

        if self.layer_Type == 'middle':
            self.a = actFun(self.z)
        elif self.layer_Type == 'last':
            exp_scores = np.exp(self.z)
            self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return None

def main():

    # # generate and visualize Make-Moons dataset
     X, y = generate_data()
     # plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
     # plt.show()

     model = NeuralNetwork(nn_input_dim=2, nn_hidden_dim=4 , nn_output_dim=2, nn_num_of_layers=4, actFun_type='sigmoid')
     model.fit_model(X,y)
     model.visualize_decision_boundary(X,y)

if __name__ == "__main__":
    main()