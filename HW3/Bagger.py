import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import time
from FeedForwardNN import NeuralNetworkClassifier
from collections import Counter


# class to wrap around bagging 5+ networks
class Bagger:
    
    def __init__(self, num_models):
        self.training_time_ = 0 # init to 0 for clarity
        self.num_models = num_models # hold on to this value
        self.nns = []

        for _ in range(num_models): # init n models
            nn = NeuralNetworkClassifier(
                n_features=dp.train_reviews.shape[1],
                hidden_layers=[450, 450], # Three hidden layers
                eta=0.0001, # Learning rate
                n_iter=1, # Epochs
                batch_size=50, # Mini-batch size
                dropout=0.0, # Dropout rate # TODO: ZEROING OUT FOR TESTING BASELINE
                random_state=42
            )
            self.nns.append(nn) # add to collection
    
    # fit the neurals with 
    def fit(self, X, y, seed=42):
        num_samples = X.shape[0] # number of samples passed
        np.random.seed(seed) # for reproducability
        random_selections = np.random.permutation(num_samples) # generate random selections
        # sanity checks
        # print(random_selections.shape)
        # print(random_selections[:10])

        training_sets = np.array_split(
            random_selections,
            self.num_models
            ) # split into parts for my sons to use

        for training_set, nn in zip(training_sets, self.nns):
            X_train = X[training_set] # grab the training indeces
            y_train = y[training_set] # grab corresponding labels
            nn.fit(X_train, y_train) # fit this model

        self.set_total_train_time() # get the total train time of all models

    # predict via majority vote from my boys
    def predict(self, X):
        majority_votes = []

        for sample in X: # for each sample
            predictions = [] # empty it out
            for nn in self.nns: # for each model
                predictions.append(nn.predict(sample)[0]) # predict this sample
            # gathered all preds, take majority vote
            
            c = Counter(predictions)
            majority_votes.append(c.most_common(1)[0][0]) # return the most common value

        return majority_votes

    # wrapper for scoring
    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)


    # sum up all the training times of the little guys
    def set_total_train_time(self):
        for nn in self.nns:
            self.training_time_ += nn.training_time_


if __name__ == "__main__":
    from DataProcessor import DataProcessor
    import matplotlib.pyplot as plt

    dp = DataProcessor()

    print(f"Training samples: {dp.train_reviews.shape[0]}")
    print(f"Test samples: {dp.test_reviews.shape[0]}")
    print(f"Feature dimension: {dp.train_reviews.shape[1]}")

    # Initialize neural network
    bagger = Bagger(8) # 8 models
    # nn = NeuralNetworkClassifier(
    #     n_features=dp.train_reviews.shape[1],
    #     hidden_layers=[256, 128, 64], # Three hidden layers
    #     eta=0.001, # Learning rate
    #     n_iter=30, # Epochs
    #     batch_size=128, # Mini-batch size
    #     dropout=0.0, # Dropout rate # TODO: ZEROING OUT FOR TESTING BASELINE
    #     random_state=42
    # )

    bagger.fit(dp.train_reviews, dp.train_sentiments)
    # nn.fit(dp.train_reviews, dp.train_sentiments)

    train_start = time.time()
    train_acc = bagger.score(dp.train_reviews, dp.train_sentiments)
    # train_acc = nn.score(dp.train_reviews, dp.train_sentiments)
    train_eval_time = time.time() - train_start

    test_start = time.time()
    test_acc = bagger.score(dp.test_reviews, dp.test_sentiments)
    # test_acc = nn.score(dp.test_reviews, dp.test_sentiments)
    test_eval_time = time.time() - test_start

    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"\nTraining Time: {bagger.training_time_:.2f} seconds") #TODO
    # print(f"\nTraining Time: {nn.training_time_:.2f} seconds") #TODO
    print(f"Train Evaluation Time: {train_eval_time:.4f} seconds")
    print(f"Test Evaluation Time: {test_eval_time:.4f} seconds")

    # Plot loss convergence
    # NOTE: commenting for this one since this plot is not as straight forward
    # plt.figure(figsize=(10, 6))
    # plt.plot(range(1, len(nn.losses_) + 1), nn.losses_, marker='o')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss (Binary Cross-Entropy)')
    # plt.title('Training Loss Convergence')
    # plt.grid(True)
    # plt.show()