#!/usr/bin/python3
# coding: utf-8
# Before execute python3 activate the following environment:
# $ source /opt/env-neupy/bin/activate

# Generalization with Self-organizing Map (SOM) and NeuPy
#
# Using a SOM to recognize handwritten digits from Optical Recognition of
# Handwritten Digits Data Set (Lichman, 2003, UCI Machine Learning Repository)

# Author: Patricio García
# Subject: Tratamiento Inteligente de Datos, ESIT-ULL
# License: GNU GPL 3
# Date: 26/04/2023

# References:
# * High-diensional data visualization: Self-Organizing Maps and Applications: http://neupy.com/2017/12/09/sofm_applications.html#high-dimensional-data-visualization
# * Generalizacion con LVQ, Patricio García: https://gitlab.com/pgarcia/OptdigitsLVQNeurolab/-/blob/master/Generalizacion_con_LVQ.ipynb

# ToDo:
# * Optimize max_iter_som

# Library imports
# ------------
# Standard scientific Python imports
# python3 -m pip install matplotlib
# python3 -m pip install neupy
# apt-get install python-tk
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Import from scikit-learn
# sudo pip install scikit-learn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

# Import from NeuPy
# pip install neupy
from neupy import algorithms, utils, init

# Import from datetime
from datetime import datetime

# Increase de figure size
plt.rcParams['figure.dpi'] = 150

# Se carga en memoria los datos, con parte de fichero csv y otra parte de librerías de datos de sklearn
# Patterns load
# ------------
# The digits dataset, train set
digits = np.loadtxt("optdigits.tra", dtype=int, delimiter=',')

(n_samples, n_features) = digits.shape

# The digits dataset, test set
digits_test = datasets.load_digits()

n_samples_test = len(digits_test.images)

print(digits_test.DESCR)

# Se visualizan las 10 clases de dígitos y se muestra en una tabla
# Data visualization
# ------------
# have a look at the first test images
n_img_plt = 10

print("Showing first %d digit images" % n_img_plt)

_, axes = plt.subplots(2, n_img_plt//2)
images_and_labels = list(zip(digits_test.images[:n_img_plt], digits_test.target[:n_img_plt]))
for ax, (image, label) in zip(np.concatenate(axes), images_and_labels):
    ax.set_axis_off()
    _ = ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    _ = ax.set_title('real: %i' % label)

plt.show()

# El 75% de los datos del primer conjunto para entrenamiento, el otro 25% para validación. Del segundo conjunto se usa el 100% para test
# No se realiza escalada de datos de entrada
# No se realiza balanceo de clases porque ya están bastante equilibradas
# Data preprocessing
# ------------
train_size = 0.75
test_size = 1 - train_size

print("Preprocessing data, %d%% for training and %d%% for validation" % (train_size * 100, test_size * 100))

# inputs and flatten images
patterns_input = digits[:, :n_features-1]
input_test = digits_test.images.reshape((n_samples_test, -1))

# targets
patterns_target = digits[:, -1]
target_test = digits_test.target

# Split and shuffle patterns
input_train, input_valid, target_train, target_valid = train_test_split(
    patterns_input, patterns_target, train_size=train_size, test_size=test_size,
    random_state=0, shuffle=True)

# Print data sets cardinality
print("Number of patterns in train set: %d" % len(target_train))
print("Number of patterns in valid set: %d" % len(target_valid))
print("Number of patterns in test set: %d" % len(target_test))
# Conjunto de entrenamiento: 2867
# Conjunto de validación: 956
# Conjunto de test: 1797

# Se entrena el modelo SOM con 55 iteraciones, una rejilla de 20x20 y distancia euclídea
# Modeling SOM
# ------------
max_iter_som = 25
grid_height = 20
grid_width = 20
distance = 'euclid'
learning_radius = 5
step = 0.5
reduce_step_after = max_iter_som - 5
std = 1.0
reduce_std_after = max_iter_som - 5
weight = init.Normal()

print("Learning %dx%d SOM with %d maximum number of iterations and ..." % (grid_height, grid_width, max_iter_som))

now = datetime.now()
# Random generator seed for NeuPy
# utils.reproducible(0)

sofm = algorithms.SOFM(
    n_inputs = input_train.shape[1],
    features_grid = (grid_height, grid_width),
    distance = distance,
    weight = weight,
    learning_radius = learning_radius,
    reduce_radius_after = max_iter_som // learning_radius,  # 0 radius at end
    step = step,
    reduce_step_after = reduce_step_after,
    std = std,
    reduce_std_after = reduce_std_after,
    shuffle_data = False,
    verbose = True,
)

sofm.train(input_train, epochs=max_iter_som)
sofm_output_train = sofm.predict(input_train)
sofm_output_valid = sofm.predict(input_valid)
print("Number of seconds for training: %d" % (datetime.now() - now).total_seconds())

# Show results
print("Visualizing the Mean Absolute Error Trajectory")
# plt.plot(range(1, len(sofm.errors.train)+1), sofm.errors.train)
plt.plot(range(1, len(sofm.errors)+1), sofm.errors)
plt.xlabel('number of iterations')
plt.ylabel('MAE')
plt.show()

# Se visualizan a continuación los prototipos de cada una de las neuronas de la rejilla SOM
# Prototypes visualization
# ------------
# have a look at the grid
def plot_prototypes_grid(grid_height, grid_width, weight, labels=[]):
    """
    Visualization prototypes of SOM grid and labels
    """
    print("Building visualization of prototypes grid ...")
    grid = gridspec.GridSpec(grid_height, grid_width)
    grid.update(wspace=0, hspace=0)
    for row_id in range(grid_height):
        print("Progress: {:.2%}".format(row_id / grid_height))
        for col_id in range(grid_width):
            index = row_id * grid_width + col_id
            sample = weight[:,index]
            _ = plt.subplot(grid[index])
            _ = plt.imshow(sample.reshape((8, 8)), cmap='Greys')
            if len(labels):
                _ = plt.text(0, 0, labels[index], color='r', fontsize=8)
            _ = plt.axis('off')
    plt.show()

plot_prototypes_grid(grid_height, grid_width, sofm.weight)
# En el mapa de prototipos se ve que casi todas las distintas clases se encuentran en regiones diferentes

# Se entrena el modelo Perceptrón Simple con 30 iteraciones
# Modeling Perceptron
# ------------
# Perceptron use de SOM output (grid array of 0 except winning output).
# That is, Counterpropagation Network (CPN)
max_iter_per = 30

print("Learning a Perceptron with %d maximum number of iterations and ..." % max_iter_per)

per = Perceptron(max_iter=max_iter_per, shuffle=False, verbose=True)
per.fit(sofm_output_train, target_train)


# Intitial results
# ------------
print("Printing initial results")

predict_train = per.predict(sofm_output_train)
predict_valid = per.predict(sofm_output_valid)

print("Train accuracy: %.3f%%" % (accuracy_score(target_train, predict_train) * 100))
print("Valid accuracy: %.3f%%" % (accuracy_score(target_valid, predict_valid) * 100))

print("Train confusion matrix:")
print(confusion_matrix(target_train, predict_train))
print("Valid confusion matrix:")
print(confusion_matrix(target_valid, predict_valid))

print("Train classification report:")
print(classification_report(target_train, predict_train))
print("Valid classification report:")
print(classification_report(target_valid, predict_valid))


# Labels visualization
# ------------
sofm_output_labels = np.zeros((grid_height * grid_width, grid_height * grid_width), dtype=int)
for i in range(grid_height * grid_width):
    sofm_output_labels[i][i] = 1
predict_labels = per.predict(sofm_output_labels)

plot_prototypes_grid(grid_height, grid_width, sofm.weight, predict_labels)
# Train accuracy: 98.151%
# Valid accuracy: 97.908%

# Se estudia las dimensiones de la rejilla SOM más adecuadas, probando con distintos valores entre 10 y 190
# Se repiten 5 veces los experimentos para cada rejilla
# Architecture optimization
# ------------
print("Architecture optimization")

# Test SOM with differents number of grid units and several repetitions
tests_grid_side = [5, 10, 15, 20, 25, 30, 35]
n_reps = 5

now = datetime.now()
best_sofm = []
best_per = []
best_acc = 0.0
accs_train = []
accs_valid = []
for grid_side in tests_grid_side:
    max_acc_train = max_acc_valid = 0.0
    for random_state in range(n_reps):
        # utils.reproducible(random_state)
        sofm = algorithms.SOFM(n_inputs = input_train.shape[1], features_grid = (grid_side, grid_side), distance = distance, weight = weight, learning_radius = learning_radius, reduce_radius_after = max_iter_som // learning_radius, step = step, reduce_step_after = reduce_step_after, std = std, reduce_std_after = reduce_std_after, shuffle_data = False, verbose = False)
        sofm.train(input_train, epochs=max_iter_som)
        sofm_output_train = sofm.predict(input_train)
        sofm_output_valid = sofm.predict(input_valid)
        per = Perceptron(max_iter=max_iter_per, shuffle=False, verbose=False)
        _ = per.fit(sofm_output_train, target_train)
        acc_train = accuracy_score(target_train, per.predict(sofm_output_train))
        acc_valid = accuracy_score(target_valid,per.predict(sofm_output_valid))
        print("Seed = %d, train acc = %.8f, valid acc = %.8f" % (random_state, acc_train, acc_valid))
        if (max_acc_valid < acc_valid):
            max_acc_valid = acc_valid
            max_acc_train = acc_train
            if (acc_valid > best_acc):
                best_acc = acc_valid
                best_per = per
                best_sofm = sofm
    accs_train.append(max_acc_train)
    accs_valid.append(max_acc_valid)
    print("Grid size = %ix%i, train acc = %.8f, max valid acc = %.8f" % (grid_side, grid_side, max_acc_train, max_acc_valid))

print("Number of seconds for training: %d" % (datetime.now() - now).total_seconds())
print("Best CPN valid accuracy: %.8f%%" % (best_acc * 100))
print("Best SOM: ", best_sofm)
print("Best Perceptron: ", best_per)

# Show results
width = 2
plt.bar(np.array(tests_grid_side) - width, 100 *(1- np.array(accs_train)), color='g', width=width, label='Train error')
plt.bar(np.array(tests_grid_side), 100 *(1- np.array(accs_valid)), width=width, label='Min valid error')
plt.xlabel('grid side')
plt.ylabel('error (%)')
plt.xticks(np.array(tests_grid_side), tests_grid_side)
plt.legend(loc='upper right')
plt.show()
# El mejor resultado es el de 20x20
# Se observa que rejillas entre 15x15 y 30x30 los resultados son similares


# Final results of best CPN
# ------------
print("Printing final results")

sofm_output_train = best_sofm.predict(input_train)
sofm_output_valid = best_sofm.predict(input_valid)
sofm_output_test = best_sofm.predict(input_test)
predict_train = best_per.predict(sofm_output_train)
predict_valid = best_per.predict(sofm_output_valid)
predict_test = best_per.predict(sofm_output_test)

print("Train accuracy: %.3f%%" % (accuracy_score(target_train, predict_train) * 100))
print("Valid accuracy: %.3f%%" % (accuracy_score(target_valid, predict_valid) * 100))
print("Test accuracy: %.3f%%" % (accuracy_score(target_test, predict_test) * 100))

print("Train confusion matrix:")
print(confusion_matrix(target_train, predict_train))
print("Valid confusion matrix:")
print(confusion_matrix(target_valid, predict_valid))
print("Test confusion matrix:")
print(confusion_matrix(target_test, predict_test))

print("Train classification report:")
print(classification_report(target_train, predict_train))
print("Valid classification report:")
print(classification_report(target_valid, predict_valid))
print("Test classification report:")
print(classification_report(target_test, predict_test))

# ROC curves of test set
per_probs = best_per.decision_function(sofm_output_test)
classes  = np.unique(target_train)
per_auc = []
per_fpr = []
per_tpr = []
for cla in classes:
   per_auc.append(roc_auc_score(target_test==cla, per_probs[:,cla]))
   fpr, tpr, _ = roc_curve(target_test==cla, per_probs[:,cla])
   per_fpr.append(fpr)
   per_tpr.append(tpr)

print("Printing ROC curves of test set")
# plot the roc curve for the model
for cla in classes:
   # plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
   _ = plt.plot(per_fpr[cla], per_tpr[cla], marker='.', label='Class %d (AUC: %.5f)' % (cla, per_auc[cla]))

# axis labels
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()
# show the plot
plt.show()
# Aunque para la clase 9 no tiene una buena curva, se ajusta rápidamente a la diagonal
# La clase 8 no tiene muy buena curva y tampoco se ajusta a la diagonal. Se podría decir que es la peor clase

# Show grid prototypes and labels
(grid_height, grid_width) = best_sofm.features_grid
sofm_output_labels = np.zeros((grid_height * grid_width, grid_height * grid_width), dtype=int)
for i in range(grid_height * grid_width):
    sofm_output_labels[i][i] = 1
predict_labels = best_per.predict(sofm_output_labels)

plot_prototypes_grid(grid_height, grid_width, best_sofm.weight, predict_labels)

# Show errors on real class 8
real_class = 8
indxs = np.where(digits_test.target == real_class)[0]
indxs_err = indxs[(np.where(predict_test[(indxs)] != real_class)[0])]
preds_err = predict_test[(indxs_err)]
n_img_plt = 8

print("Showing first %d errors of real class %d" % (n_img_plt, real_class))

_, axes = plt.subplots(2, n_img_plt//2)
images_and_labels = list(zip(digits_test.images[(indxs_err)], digits_test.target[(indxs_err)], preds_err))
for ax, (image, label1, label2) in zip(np.concatenate(axes), images_and_labels):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title('real:%i pred:%i' % (label1, label2))

plt.show()
# Los problemas que se aprecia que con el dígito 8 resultas ser lógicos (No se puede distinguir bien qué número es)