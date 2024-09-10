import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
import pandas as pd
from gd_utils import *

selected_classes = [0, 9]           # selecting 2 classes (digits) to make it binary
NUM_EPOCHS = 5


# Load and preprocess data
mnist = fetch_openml('mnist_784')

df = pd.DataFrame.from_dict({'data': list(mnist['data'].astype('float64').values), 'target': mnist['target'].astype('int')})

# filter by class, shuffle, divide to train/validation/test
df = df.loc[df['target'].isin(selected_classes)]
df['target'] = df['target'].replace(to_replace=selected_classes[0], value=0)
df['target'] = df['target'].replace(to_replace=selected_classes[1], value=1)

df = df.sample(frac=1).reset_index(drop=True)               # Need to add a constant seed for reproduction
data_split = [0.6, 0.2, 0.2]        # train/validation/test

num_samples = len(df)
df_train = df.iloc[:int(num_samples * data_split[0]), :]
df_validation = df.iloc[int(num_samples * data_split[0]): int(num_samples * (data_split[1] + data_split[0])), :]
df_test = df.iloc[int(num_samples * (data_split[1] + data_split[0])):, :]

train_data = np.stack(df_train['data'].to_numpy())
validation_data = np.stack(df_validation['data'].to_numpy())

# Adding a bias term
train_data = np.concatenate((train_data, np.ones((len(train_data), 1), dtype='float64')), axis=1)
validation_data = np.concatenate((validation_data, np.ones((len(validation_data), 1), dtype='float64')), axis=1)

train_data = train_data.transpose() / 255.
validation_data = validation_data.transpose() / 255.

train_targets = df_train['target'].to_numpy()
validation_targets = df_validation['target'].to_numpy()


# model
# initialize weights
weight_length = 785  # +1 for the bias term
weights = np.random.uniform(size=(1, weight_length))
weights[-1] = 1  # Bias term

# Hyperparameters
learning_rate = 0.01

# For plotting
training_losses = []
validation_accuracies = []

# Training loop
previous_accuracy = 0
for epoch in range(NUM_EPOCHS):
    output = np.dot(weights, train_data)
    logits = sigmoid(output)
    epoch_loss = bce_loss(logits, train_targets)
    grads = bce_grad(logits, train_targets, train_data)

    weights = update_weights_vanilla(weights, grads, learning_rate)

