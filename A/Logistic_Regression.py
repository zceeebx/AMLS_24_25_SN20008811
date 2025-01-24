import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

def add_intercept(x):
    """
    Adds a column of ones (intercept) to the input feature matrix.

    Parameters:
    - x (ndarray): The input feature matrix of shape (n_samples, n_features).

    Returns:
    - ndarray: The input feature matrix with an added column of ones (intercept term).
    """
    # Create a column of ones with the same number of rows as 'x'
    intercept = np.ones((x.shape[0], 1))

    # Concatenate the intercept column with the input feature matrix 'x'
    return np.concatenate((intercept, x), axis=1)

def sigmoid(z):
    """
    Computes the sigmoid function for the input value(s).

    Parameters:
    - z (ndarray or scalar): The input value(s) for which the sigmoid function is calculated.

    Returns:
    - ndarray or scalar: The result of applying the sigmoid function element-wise to `z`.
    """
    # Clip the values of z to avoid overflow when calculating exp(-z)
    z = np.clip(z, -500, 500)

    # Apply the sigmoid function formula: sigmoid(z) = 1 / (1 + exp(-z))
    return 1. / (1. + np.exp(-z))

def compute_loss(y, h):
    """
    Computes the binary cross-entropy (log loss) loss function.

    Parameters:
    - y (ndarray): True labels (0 or 1) for each sample in the dataset.
    - h (ndarray): Predicted probabilities for each sample, typically from a sigmoid function.

    Returns:
    - float: The calculated binary cross-entropy loss.
    """
    # constrain the h value in the range (epsilon, 1-epsilon)
    epsilon = 1e-15  
    # Clip the predicted probabilities (h) to avoid log(0) and overflow errors
    h = np.clip(h, epsilon, 1 - epsilon)
    
    # Compute the binary cross-entropy loss
    return -np.mean(y * np.log2(h) + (1 - y) * np.log2(1 - h))

def evaluate_loss_from_val(xVal, yVal, theta):
    """
    Evaluates the loss on the validation dataset using the current model parameters (theta).

    Parameters:
    - xVal (ndarray): The feature matrix for the validation dataset.
    - yVal (ndarray): The true labels for the validation dataset.
    - theta (ndarray): The current model parameters (weights) used to make predictions.

    Returns:
    - float: The calculated binary cross-entropy loss on the validation set.
    """
    # Add intercept (bias) term to the validation features
    xVal = add_intercept(xVal)
    
    # Compute the linear combination of inputs and weights (z = xVal * theta)
    z = np.dot(xVal, theta)
    
    # Apply the sigmoid function to get the predicted probabilities (h)
    h = sigmoid(z)
    
    # Compute and return the loss using the true labels (yVal) and predicted probabilities (h)
    return compute_loss(yVal, h)

def predict_y(x, theta):
    """
    Makes predictions on the input data (x) using the logistic regression model with parameters (theta).

    Parameters:
    - x (ndarray): The feature matrix for the input data (each row represents a data point).
    - theta (ndarray): The model parameters (weights), including the intercept term.

    Returns:
    - ndarray: The predicted class labels (0 or 1) for each data point.
    """
    # Add intercept (bias) term to the input features
    x = add_intercept(x)
    
    # Compute the linear combination of inputs and weights (z = x * theta)
    z = np.dot(x, theta)
    
    # Apply sigmoid to get the predicted probabilities (h)
    y_pred = sigmoid(z)
    
    # Return the class predictions (1 if probability >= 0.5, else 0)
    return (y_pred >= 0.5).astype(int)

def gradient_descent(xTrain, yTrain, lr, max_iter=1000, patience=10):
    """
    Performs gradient descent to optimize the logistic regression model parameters (theta).

    Parameters:
    - xTrain (ndarray): The input feature matrix for the training data (each row is a data point).
    - yTrain (ndarray): The true labels for the training data (binary labels: 0 or 1).
    - lr (float): The learning rate used to update the model parameters.
    - max_iter (int): The maximum number of iterations for the gradient descent process.
    - patience (int): The number of iterations without improvement in loss before early stopping.

    Returns:
    - ndarray: The optimized model parameters (theta).
    """
    # Add intercept (bias) term to the input features
    xTrain = add_intercept(xTrain)
    
    # Initialize parameters (theta) to zeros
    theta = np.zeros(xTrain.shape[1])
    
    # Set initial conditions for early stopping
    lowest_loss = float('inf')
    no_improve_count = 0

    # Perform gradient descent for a specified number of iterations (max_iter)
    for i in range(max_iter):
        z = np.dot(xTrain, theta)   # Compute the linear combination of inputs and model parameters (z = x * theta)
        h = sigmoid(z)              # Apply the sigmoid function to get predicted probabilities (h)
        gradient = np.dot(xTrain.T, (h - yTrain)) / yTrain.shape[0]     # Calculate the gradient of the loss function with respect to the model parameters
        theta -= lr * gradient      # Update model parameters using the learning rate and gradient

        # Calculate the current loss
        loss = compute_loss(yTrain, h)
        
        # Early stopping: If the loss improves, reset the no improvement counter
        if loss < lowest_loss:
            lowest_loss = loss
            no_improve_count = 0
        else:   
            # If loss doesn't improve, increase the counter
            no_improve_count += 1
            if no_improve_count >= patience:
                # Stop early if the loss hasn't improved for a specified number of iterations
                print(f"Early stopping at iteration {i}")
                break
    
    # Return the optimized model parameters (theta)
    return theta

def cross_validation(x, y, k, learning_rates):
    """
    Perform k-fold cross-validation to find the best learning rate for logistic regression.

    Parameters:
    - x (ndarray): The input feature matrix for the dataset (each row is a data point).
    - y (ndarray): The target labels for the dataset.
    - k (int): The number of folds in cross-validation.
    - learning_rates (list of float): A list of learning rates to evaluate.

    Returns:
    - best_lr (float): The learning rate that resulted in the lowest average validation loss.
    """
    # Initialize k-fold cross-validation and the dictionary to store validation losses for each learning rate
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    val_losses = {lr: [] for lr in learning_rates}
    lowest_val_loss = float('inf')
    best_lr = None

    # Iterate over each learning rate to evaluate its performance
    for lr in learning_rates:
        fold_losses = []
        for train_idx, val_idx in kf.split(x):
            # Split the data into training and validation sets for each fold
            x_train, x_val = x[train_idx], x[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Train the model using gradient descent with the current learning rate
            theta = gradient_descent(x_train, y_train, lr)

            # Evaluate the loss on the validation set
            val_loss = evaluate_loss_from_val(x_val, y_val, theta)
            fold_losses.append(val_loss)

        # Compute the average validation loss for this learning rate
        avg_val_loss = np.mean(fold_losses)
        val_losses[lr] = fold_losses  # Store fold losses for this learning rate
        print(f"Learning rate: {lr}, Average validation loss: {avg_val_loss:.4f}")

        # Update the best learning rate if the current one has the lowest average validation loss
        if avg_val_loss < lowest_val_loss:
            lowest_val_loss = avg_val_loss
            best_lr = lr
        
        # Plot the validation losses for each fold for the current learning rate
        plt.plot(range(k), val_losses[lr], marker='o', label=f"LR = {lr}")

    # Find the lowest loss and its index for the best learning rate
    min_loss = min(val_losses[lr])  
    min_loss_index = val_losses[lr].index(min_loss)  

    # Annotate the plot with the lowest loss point
    plt.annotate(f'({min_loss_index}, {min_loss:.4f})', 
    xy=(min_loss_index, min_loss), 
    xytext=(min_loss_index + 1, min_loss + 0.05),  
    arrowprops=dict(facecolor='red', arrowstyle="->"),
    fontsize=9, color='red')

    # Print the best learning rate and its corresponding lowest validation loss
    print(f"Best learning rate: {best_lr}, Lowest average validation loss: {lowest_val_loss:.4f}")
    
    # Plot learning rate vs validation loss curve
    plt.xlabel('Fold')
    plt.ylabel('Validation Loss')
    plt.title('K-Fold Cross Validation Loss')
    plt.legend()
    plt.show()

    # Return the best learning rate
    return best_lr

def lr_search (x_train, y_train, x_val, y_val, learning_rates):
    """
    Perform a search for the best learning rate by training a logistic regression model
    and evaluating its performance on the validation set for different learning rates.

    Parameters:
    - x_train (ndarray): The input features for the training set.
    - y_train (ndarray): The target labels for the training set.
    - x_val (ndarray): The input features for the validation set.
    - y_val (ndarray): The target labels for the validation set.
    - learning_rates (list of float): A list of learning rates to evaluate.

    Returns:
    - best_lr (float): The learning rate that results in the lowest validation loss.
    """
    # Initialize the dictionary to store validation losses for each learning rate
    best_lr = None
    lowest_val_loss = float('inf')
    val_losses = []

    # Iterate over all the learning rates provided
    for lr in learning_rates:
        # Train the logistic regression model with the current learning rate
        theta = gradient_descent(x_train, y_train, lr)
        # Evaluate the model's performance on the validation set using the current learning rate
        val_loss = evaluate_loss_from_val(x_val, y_val, theta)
        # Append the validation loss for this learning rate to the list
        val_losses.append(val_loss)
        
        # Print the validation loss for the current learning rate
        print(f"Learning rate: {lr}, Validation loss: {val_loss:.4f}")
        
        # Update the best learning rate if the current one gives a lower validation loss
        if val_loss < lowest_val_loss:
            lowest_val_loss = val_loss
            best_lr = lr

    # Print the best learning rate and its corresponding validation loss
    print(f"Best learning rate: {best_lr}, Lowest validation loss: {lowest_val_loss:.4f}")

    # Plot the learning rates versus their corresponding validation losses
    plt.plot(learning_rates, val_losses, marker='o')
    # Find the minimum loss and its index to annotate on the plot
    min_loss = min(val_losses)  
    min_loss_index = val_losses.index(min_loss)  

    # Annotate the plot with the best learning rate (lowest validation loss)
    plt.annotate(f'({learning_rates[min_loss_index]}, {min_loss:.4f})', 
    xy=(learning_rates[min_loss_index], min_loss), 
    xytext=(learning_rates[min_loss_index] + 0.001, min_loss + 0.05),  # 调整文本位置
    arrowprops=dict(facecolor='red', arrowstyle="->"),
    fontsize=9, color='red')
    
    # Configure and display the plot
    plt.xlabel('Learning Rate')
    plt.ylabel('Validation Loss')
    plt.title('Learning Rate vs Validation Loss')
    plt.show()
    
    # Return the best learning rate
    return best_lr

def logistic_regression_classifier(x_train, y_train, x_val, y_val, x_test, y_test, x_train_full, y_train_full, x_test_cross, k):
    """
    Train a logistic regression classifier, either with or without cross-validation, and evaluate the model's performance.

    Parameters:
    - x_train (ndarray): The training input features.
    - y_train (ndarray): The training target labels.
    - x_val (ndarray): The validation input features.
    - y_val (ndarray): The validation target labels.
    - x_test (ndarray): The test input features (for final evaluation).
    - y_test (ndarray): The test target labels.
    - x_train_full (ndarray): The full training set for cross-validation.
    - y_train_full (ndarray): The full training labels for cross-validation.
    - x_test_cross (ndarray): The test set for cross-validation evaluation.
    - k (int): Determines whether to use cross-validation (k > 0) or not (k = 0).

    Returns:
    - y_pred (ndarray): The predicted labels for the test set.
    """
    
    # Hyperparameter tuning for learning rate (using a fixed list of learning rates)
    learning_rates = [0.001, 0.003, 0.01, 0.1]
    
    # If no cross-validation (k = 0)
    if k == 0:
        # Search for the best learning rate using `lr_search` function
        best_lr = lr_search (x_train, y_train, x_val, y_val, learning_rates)
        # Train the model using the best learning rate and gradient descent
        final_theta = gradient_descent(x_train, y_train, best_lr)
        # Predict on the test set
        y_pred = predict_y(x_test, final_theta)
    # If cross-validation is used (k > 0)
    elif k > 0:
        # Search for the best learning rate using `cross_validation` function
        best_lr = cross_validation(x_train_full, y_train_full, k, learning_rates)
        # Train the model using the best learning rate and gradient descent
        final_theta = gradient_descent(x_train_full, y_train_full, best_lr)
        # Predict on the test set for cross-validation
        y_pred = predict_y(x_test_cross, final_theta)
    # If an invalid value for k is provided
    else:
        raise ValueError("Invalid k value.")

    # Evaluate the model performance on the test set
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(conf_matrix)
    print(f"Accuracy on test set: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred, zero_division=1))
    
    # Visualize the confusion matrix using seaborn heatmap
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    # Title based on whether cross-validation was used or not
    if k == 0:
        plt.title(f"Best Logistic Regression without Cross-Validation Confusion Matrix")
    elif k > 0:
        plt.title(f"Best Logistic Regression with Cross-Validation Confusion Matrix")
    else:
        raise ValueError("Invalid k value.")
    
    # Labels for the confusion matrix plot
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()      # Show the plot
    
    # Return the predicted labels for the test set
    return y_pred
