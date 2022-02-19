# Coded in Python and takes some functions from numpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import random as rand
from sklearn.model_selection import train_test_split
from keras.datasets import mnist

# Code to plot the 10 images (from mnist)
(x_train, y_train), (x_test, y_test) = mnist.load_data()    #separate train and test data; unpacks a data set that was specifically pickled into a format
images = 10;        # 10 images 
plt.figure(figsize=(images*2,2))      #adjust size of the figure object.h
plt.gray()          #sets color map to gray
for i in range(images):
  plt.subplot(1, images, i+1)     # (nrows, ncols, index)
  plt.imshow(x_train[i])            #displays data as an image
plt.show()                          #displays all figures 
# To get images in black and white
x_train = x_train/255;

def SSL(M = 1, K = 5, alpha = 100, ratio = 50):
  #M is ..., K is number of nearest neighbors, alpha is how much of the ground truth was used in label propagation, ratio is % of nodes labeled 
  rand.seed(1)
  #ratio = float(ratio/100)
  #alpha = float(alpha/100)
  nodes = 500;                  # Assign 500 nodes
  X_images = x_train[:nodes]
  X_labels = y_train[:nodes]

  ratio = 0.2; #assuming 20% is labeled 
  labeled_amnt = int(np.floor(ratio*nodes)); #amount of nodes we want labeled, np.floor rounds the number down and gives floor value

  # One hot encoding is converting categorical data variables to be provided to machine learning algorithms to improve prediction
  # assigns binary numbers to represent the data
  Y = np.zeros((nodes, 10)) #100 by 10 (one hot vector)

  # Randomizes the index
  random_index = np.random.choice(nodes, labeled_amnt, replace=False)

  # For Loop to randomize the indexes with labels of 1
  for indexes in range(labeled_amnt):
    index = random_index[indexes]; 
    curr_label = X_labels[index]; 
    Y[index][curr_label] = 1; 

  # Gamma value is 1/2σ^2(where is gamma in the paper?)
  g = 1.25; 

  # Find the distances of the images 
  # np.zeroes gives an array filled with zeroes (row, column)
  W_matrix = np.zeros((nodes, nodes))
  distances = np.zeros((nodes, nodes))
  for i in range(nodes):
    for j in range(nodes):
      distances[i][j] = np.linalg.norm(X_images[i] - X_images[j]);
      d1 = distances[i][j]
      W_matrix[i][j] = np.exp(-g*d1**2)

  # If the row equals the matrix, label it as 0 ? Why ?
  for i in range(nodes):
    for j in range(nodes):
      if i == j:
        W_matrix[i][j] = 0

  max_iter = 20;
  alpha = 0.8;
  K = 5;

  # Graphing the KNN graph and sorts the distances by closeness 
  knn_graph = np.zeros((nodes, nodes))
  for node in range(nodes):
    nodes_closest = np.argsort(distances[node])
    for closest in range(K): 
      index = nodes_closest[closest+1]
      knn_graph[node][index] = W_matrix[node][index];   #what was the point of this again
      knn_graph[index][node] = W_matrix[node][index];

  knn_degrees = np.zeros((1, nodes))
  for i in range(nodes):
    knn_degrees[0][i] = np.count_nonzero(knn_graph[i]);

  M = 30      #M is the number of highest/lowest degrees 
  lowestM_degs = np.argsort(knn_degrees[0])[:M];
  highestM_degs = np.argsort(-1*knn_degrees[0])[:M];

  Dmatrix = np.zeros((nodes, nodes))
  for i in range(nodes):
    Dmatrix[i][i] = np.sum(knn_graph[i])

  Dinv = np.linalg.inv(Dmatrix)
  S = Dinv@knn_graph

  # Finding the iteration value (F)
  F = Y;
  for i in range(max_iter):
    F = alpha * S @ F + (1 - alpha) * Y

  # Understand this part more ....
  Y_final = np.zeros(nodes)       #what is Y_final
  for i in range(nodes):
    label = np.argmax(F[i]);  #np.argmax returns the indices of the maximum values along an axis.
    Y_final[i] = label

  correct = 0
  for val in range(nodes):
    if X_labels[val] == Y_final[val]:
      correct = correct + 1

  noflip = correct*100/nodes

  #flips lowest M degree nodes to random label
  for i in range(len(lowestM_degs)):
    while Y_final[lowestM_degs[i]] == X_labels[lowestM_degs[i]]: 
      Y_final[lowestM_degs[i]] = rand.randint(0,9);

  correct = 0
  for val in range(nodes):
    if X_labels[val] == Y_final[val]:
      correct = correct + 1

  lowest = correct*100/nodes

  #flips highest M degree nodes to random label
  for i in range(len(highestM_degs)):
    while Y_final[highestM_degs[i]] == X_labels[highestM_degs[i]]:
      Y_final[highestM_degs[i]] = rand.randint(0,9);

  correct = 0
  for val in range(nodes):
    if X_labels[val] == Y_final[val]:
      correct = correct + 1

  highest = correct*100/nodes

  print("Accuracy:", correct*100/nodes)
  return noflip, lowest, highest

Mvalues = [5, 10, 15, 20, 30, 40, 50, 75, 100, 125, 150, 200, 300]

NoFlip_Accuracy = np.zeros(len(Mvalues))
LowFlip_Accuracy = np.zeros(len(Mvalues))
HighFlip_Accuracy = np.zeros(len(Mvalues))
for M in range(len(Mvalues)):
  [noflip, lowest, highest] = SSL(M = Mvalues[M])
  NoFlip_Accuracy[M] = noflip
  LowFlip_Accuracy[M] = lowest
  HighFlip_Accuracy[M] = highest
plt.plot(Mvalues, NoFlip_Accuracy)
plt.plot(Mvalues, LowFlip_Accuracy)
plt.plot(Mvalues, HighFlip_Accuracy)
plt.legend({'No Flipping','Low Flipping','High Flipping'})
plt.xlabel("M Values")
plt.ylabel("Accuracy")
plt.show() 


