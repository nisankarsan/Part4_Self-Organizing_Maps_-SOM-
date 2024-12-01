#   Mega Case Study - Make a hybrid Deep Learning Model


# Part 1 - Identify the Frauds with the Self Organization Map
# Self Organizing Map

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#class =0 apllication is not approve, 1 is approved
# Importing the dataset
dataset = pd.read_csv('Credit_Card_Applications.csv')

X= dataset.iloc[:,:-1].values #all columns except last one 
y= dataset.iloc[:,-1].values #last column 

#   Feature Scaling 
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)

#   Training the SOM
from minisom import MiniSom
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)

#Visualizing the result
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()

#   Finding the frauds
mappings = som.win_map(X)
frauds = np.concatenate((mappings[(4,1)], mappings[(1,8)]), axis = 0 )
frauds = sc.inverse_transform(frauds)




# Part 2 - Going from Unsupervised to Supervised Deep Learning 

#   Creating the matrix of features

customers = dataset.iloc[:,1:].values  #all columns except first one 

#   Creating the dependent variable

is_fraud =  np.zeros(len(dataset))
for i in range(len(dataset)):
    if dataset.iloc[i,0] in frauds:
        is_fraud[i] = 1

#   fFature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
customers = sc.fit_transform(customers)

#   Part 2 - make the ANN

#   Importing the Keras libraries and packaging 
from keras.models import Sequential
from keras.layers import Dense


#Initialasing the ANN
classifier = Sequential()

#   Adding the input layer and the first hidden layer
classifier.add(Dense(units = 2, kernel_initializer = 'uniform', activation = 'relu', input_dim = 15))

#output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

#   Compiling the ANN

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#   Fitting the ANN to the Training set
classifier.fit(customers, is_fraud, batch_size = 1, epochs = 2)

#   Part 3 - Making predictions and evaluating the model

#   predicting the probabilities of rauds
y_pred = classifier.predict(customers)
y_pred = np.concatenate((dataset.iloc[:, 0:1].values, y_pred), axis = 1 )
y_pred = y_pred[y_pred[:, 1].argsort()]

