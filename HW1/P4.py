import numpy as np


def fit_mini_batch_sgd(self, X, y, batch_size=32, shuffle=True):

    # Make some random numbers to work with
    rgen = np.random.RandomState(self.random_state)

    # Absorb bias column to X
    X_with_bias = np.c_[np.ones(X.shape[0]), X]
    n_examples = X_with_bias.shape[0]

    # Initialize weights
    self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X_with_bias.shape[1])
    self.losses_ = []

    # Iterate through epochs
    for epoch in range(self.n_iter):

        # Everything i looked up said to shuffle the data so that the model
        # doesn't see the training data in the same order
        if shuffle:
            indices = np.random.permutation(n_examples)
            X_shuffled = X_with_bias[indices]
            y_shuffled = y[indices]
        else:
            X_shuffled = X_with_bias
            y_shuffled = y

        epoch_losses = []

        # Break up the loop by mini batch
        for start_idx in range(0, n_examples, batch_size):

            # Calculate end index for this batch depends on the size of our inputs
            # to this method
            end_idx = min(start_idx + batch_size, n_examples)

            # Extract a batch
            X_batch = X_shuffled[start_idx:end_idx]
            y_batch = y_shuffled[start_idx:end_idx]
            batch_actual_size = len(y_batch)

            # Predictions
            net_input = self.net_input(X_batch)
            output = self.activation(net_input)

            # true_labels - predictions
            errors = y_batch - output

            # Calculate gradient and update weights
            gradient = X_batch.T.dot(errors) / batch_actual_size
            self.w_ += self.eta * 2.0 * gradient

            # Calculate loss for this batch
            batch_loss = (-y_batch.dot(np.log(output)) -
                          (1 - y_batch).dot(np.log(1 - output))) / batch_actual_size
            epoch_losses.append(batch_loss)

        # Track the loss
        self.losses_.append(np.mean(epoch_losses))

    return self