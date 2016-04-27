import numpy as np
import speaker_recognizer
from six.moves import cPickle as pickle

def tanh(x):
    return np.tanh(x)

def tanh_deriv(x):
    return 1.0 - np.tanh(x)**2

def logistic(x):
    return 1/(1 + np.exp(-x))

def logistic_derivative(x):
    return logistic(x)*(1-logistic(x))


class NeuralNetwork:
    def __init__(self, layers, activation='tanh'):
        """
        :param layers: A list containing the number of units in each layer.
        Should be at least two values
        :param activation: The activation function to be used. Can be
        "logistic" or "tanh"
        """
        if activation == 'logistic':
            self.activation = logistic
            self.activation_deriv = logistic_derivative
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_deriv = tanh_deriv

        self.weights = []
        for i in range(1, len(layers) - 1):
            self.weights.append((2*np.random.random((layers[i - 1] + 1, layers[i] + 1))-1)*0.25)
        self.weights.append((2*np.random.random((layers[i] + 1, layers[i +
                            1]))-1)*0.25)
    def fit(self, X, y, learning_rate=0.2, epochs=250000):
        X = np.atleast_2d(X)
        temp = np.ones([X.shape[0], X.shape[1]+1])
        temp[:, 0:-1] = X  # adding the bias unit to the input layer
        X = temp
        y = np.array(y)

        for k in range(epochs):
            iterator = k % 5000
            iterator = np.random.randint(X.shape[0])
            a = [X[iterator]]


            # i = np.random.randint(X.shape[0])
            # a = [X[i]]

            for l in range(len(self.weights)):
                a.append(self.activation(np.dot(a[l], self.weights[l])))

            error = y[iterator] - a[-1]
            # error = y[i] - a[-1]
            deltas = [error * self.activation_deriv(a[-1])]

            for l in range(len(a) - 2, 0, -1): # we need to begin at the second to last layer
                deltas.append(deltas[-1].dot(self.weights[l].T)*self.activation_deriv(a[l]))
            deltas.reverse()
            for i in range(len(self.weights)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * layer.T.dot(delta)


    def predict(self, x):
            x = np.array(x)
            temp = np.ones(x.shape[0]+1)
            temp[0:-1] = x
            a = temp
            for l in range(0, len(self.weights)):
                a = self.activation(np.dot(a, self.weights[l]))
            return a

    def save(self, file_name):
        f = open(file_name, 'wb')
        pickle.dump(self.weights, f, pickle.HIGHEST_PROTOCOL)
        f.close()

    def load(self, file_name):
        f = open(file_name, 'r')
        self.weights = pickle.load(f)
        f.close()





################################original trial 1
# X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# y = np.array([0, 1, 1, 0])
# for i in [[0, 0], [0, 1], [1, 0], [1,1]]:
#     print(i,nn.predict(i))
##########end of origianl trial

# ################################3trail 1 0to7 prediction
# nn = NeuralNetwork([3,8,8,8], 'tanh')
# X = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], \
#               [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1] \
#               ])
# yRaw = np.array([0, 1, 2, 3, 4, 5, 6, 7])
#
# for i in yRaw:
#     temp = np.zeros((1,len(yRaw)))
#     temp[0,i] = 1
#     # y.append(temp)
#     # yArr = np.array(temp)
#     if i == 0:
#         y = temp
#     else:
#         y = np.vstack((y,temp))
# y = np.array(y)
# nn.fit(X, y)
# for i in [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], \
#           [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1] \
#           ]:
#     print(i, nn.predict(i))
#     ######################end of trail 1
# y = []
#

class_names, train_dataset, train_labels = speaker_recognizer.model_preprocess('training_data')
class_test, test_dataset, test_labels = speaker_recognizer.model_preprocess('testing_data')
########start of trail 2 #################
#nn = NeuralNetwork([400,25,25,10], 'tanh')########[400,25,25,10]and 250000 works perfect
nn = NeuralNetwork([13, 25, 25, 3], 'tanh')
#Xorigin = np.genfromtxt('D:\pyCharmWorkSpace\Xinput.txt', delimiter=',')
Xorigin = train_dataset
# # X1 = Xorigin[:10]
# # X2 = Xorigin[500:510]
# # X = np.vstack((X1,X2))
# #
# # Xtest = Xorigin[:10]
#yOrigin = np.genfromtxt('D:\pyCharmWorkSpace\yCheck.txt', delimiter=',')
yOrigin = train_labels
# y1 = yOrigin[:10]
# y2 = yOrigin[500:510]
# y = np.hstack((y1, y2))
count = 0
for i in yOrigin:
    temp = np.zeros((1,3))
    temp[0,int(i)] = 1
    # y.append(temp)
    # yArr = np.array(temp)
    if count == 0:
        y = temp
    else:
        y = np.vstack((y,temp))
    count += 1
nn.fit(Xorigin, y)
#
random = np.random.randint(0,4000,(1,50))

yAnswer = test_labels
Xtest = test_dataset


'''
for i in range(len(random[0])):
    if i == 0:
        yAnswer = y[random[0,0]]
        Xtest = Xorigin[random[0,0]]
    else:
        yAnswer = np.vstack((yAnswer,y[random[0,i]]))
        Xtest = np.vstack((Xtest,Xorigin[random[0,i]]))
        '''


for i in range(len(Xtest)):
    print(nn.predict(Xtest[i]),'answer is :\n',yAnswer[i])

'''for i in range(len(nn.weights)):
    if i == 0:
        np.savetxt('weight0.csv', nn.weights[0], delimiter=',')
    if i == 1:
        np.savetxt('weight1.csv', nn.weights[1], delimiter=',')
    if i == 2:
        np.savetxt('weight2.csv', nn.weights[2], delimiter=',')
    if i == 3:
        np.savetxt('weight3.csv', nn.weights[3], delimiter=',')
        '''
