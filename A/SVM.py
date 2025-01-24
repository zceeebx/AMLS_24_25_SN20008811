import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def train_and_select_model(x_train, y_train, x_val, y_val, kernels, C_values):
    """
    Trains multiple SVM models with different kernels and C values, and selects the best model based on validation accuracy.

    Parameters:
    - x_train (ndarray): The training data features.
    - y_train (ndarray): The training data labels.
    - x_val (ndarray): The validation data features.
    - y_val (ndarray): The validation data labels.
    - kernels (list): A list of SVM kernel types to evaluate (e.g., ['linear', 'rbf', 'poly']).
    - C_values (list): A list of regularization parameters (C) to evaluate.

    Returns:
    - best_model (SVC object): The best trained SVM model.
    - best_params (dict): The parameters (`kernel` and `C`) for the best model.
    - val_accuracies (dict): A dictionary containing validation accuracies for each kernel.
    """
    
    # Initialize variables to track the best model and its parameters
    best_model = None
    best_accuracy = 0
    best_params = {}
    # Dictionary to store validation accuracies for each kernel
    val_accuracies = {kernel: [] for kernel in kernels}

    # Iterate over the different kernels
    for kernel in kernels:
        print(f"\nKernel: {kernel}")
        
        # Iterate over the different C values
        for C in C_values:
            # Initialize the model with the current kernel and C
            model = SVC(kernel=kernel, C=C, class_weight='balanced')
            # Train the model on the training data
            model.fit(x_train, y_train)
            # Predict on the validation set
            y_val_pred = model.predict(x_val)
            # Calculate the validation accuracy
            val_accuracy = accuracy_score(y_val, y_val_pred)
            # Store the validation accuracy for the current kernel
            val_accuracies[kernel].append(val_accuracy)

            # Print the current kernel, C, and validation accuracy
            print(f"C: {C}, validation accuracy: {val_accuracy:.4f}")

            # Update the best model if the current model has better validation accuracy
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_model = model
                best_params = {'kernel': kernel, 'C': C}

    # Return the best model, its parameters, and the validation accuracies for each kernel
    return best_model, best_params, val_accuracies

def evaluate_model(model, x_test, y_test):
    """
    Evaluates the performance of a trained model on the test set and prints various performance metrics.

    Parameters:
    - model: The trained machine learning model to be evaluated.
    - x_test (ndarray): The test data features.
    - y_test (ndarray): The test data labels.

    Returns:
    - conf_matrix (ndarray): The confusion matrix of the model's predictions on the test set.
    """
    
    # Predicting the test set using the trained model
    y_test_pred = model.predict(x_test)
    # Calculating the accuracy of the model on the test set
    test_accuracy = accuracy_score(y_test, y_test_pred)
    # Generating the confusion matrix for the test set
    conf_matrix = confusion_matrix(y_test, y_test_pred)
    # Generating the classification report for the test set
    classification_rep = classification_report(y_test, y_test_pred)

    # Printing the test accuracy and classification report
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(classification_rep)

    # Returning the confusion matrix
    return conf_matrix

def plot_validation_accuracy(kernels, C_values, val_accuracies):
    """
    Plots the validation accuracy for different kernel types and C values in Support Vector Classifiers.

    Parameters:
    - kernels (list): List of SVM kernel types to evaluate (e.g., ['linear', 'rbf']).
    - C_values (list): List of regularization parameters (C values) to be used in SVM.
    - val_accuracies (dict): A dictionary where keys are kernel types, and values are lists of validation accuracies corresponding to each C value.
    """
    # Create subplots for each kernel type
    fig, axs = plt.subplots(len(kernels), 1, figsize=(18, 5))
    for i, kernel in enumerate(kernels):
        # Plot validation accuracies for each kernel against C values
        axs[i].plot(C_values, val_accuracies[kernel], marker='o')
        
        # Find the maximum validation accuracy and its corresponding C value
        max_accuracy = max(val_accuracies[kernel])  
        max_accuracy_index = val_accuracies[kernel].index(max_accuracy)  

        # Annotate the maximum accuracy point on the plot
        axs[i].annotate(f'({C_values[max_accuracy_index]}, {max_accuracy:.4f})', 
                        xy=(C_values[max_accuracy_index], max_accuracy), 
                        xytext=(C_values[max_accuracy_index] * 1.1, max_accuracy - 0.05),  # 调整文本位置
                        arrowprops=dict(facecolor='green', arrowstyle="->"),
                        fontsize=9, color='green')
        
        # Set plot title, x and y labels
        axs[i].set_title(f"{kernel} Kernel - Validation Accuracy")
        axs[i].set_xlabel("C Value")
        axs[i].set_ylabel("Validation Accuracy")
        # Set x-axis to logarithmic scale for better visualization
        axs[i].set_xscale("log")
        
    # Adjust layout to avoid overlap and display the plot
    plt.tight_layout()
    plt.show()
    
def plot_confusion_matrix(conf_matrix, best_params):
    """
    Plots the confusion matrix using a heatmap for the best SVM model.

    Parameters:
    - conf_matrix (ndarray): Confusion matrix for the model's predictions.
    - best_params (dict): Dictionary containing the best model's parameters (e.g., kernel type).
    """
    # Create a figure with specified size
    plt.figure(figsize=(6,6))
    
    # Use seaborn's heatmap to display the confusion matrix
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    # Title that indicates the kernel type of the best model
    plt.title(f"Best SVM Model ({best_params['kernel']} Kernel) Confusion Matrix")
    # Label for x-axis (Predicted Labels) and y-axis (True Labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    # Display the plot
    plt.show()

def SVM(x_train, y_train, x_val, y_val, x_test, y_test):
    """
    Trains and evaluates the best SVM model using various kernels and hyperparameters.

    Parameters:
    - x_train (ndarray): Feature matrix for the training data.
    - y_train (ndarray): Labels for the training data.
    - x_val (ndarray): Feature matrix for the validation data.
    - y_val (ndarray): Labels for the validation data.
    - x_test (ndarray): Feature matrix for the test data.
    - y_test (ndarray): Labels for the test data.

    Returns:
    - best_model (SVC): The best trained SVM model with the highest validation accuracy.
    - best_params (dict): Dictionary containing the best hyperparameters (kernel type and C value).
    - val_accuracies (dict): Dictionary containing validation accuracies for each kernel and its associated C values.
    - conf_matrix (ndarray): Confusion matrix for the model's predictions on the test set.
    """
    # Define the possible kernels and C values to try during hyperparameter tuning
    kernels = ['linear', 'rbf', 'poly']
    C_values = [0.001, 0.01, 0.1, 1, 10]

    # Call function to train and select the best SVM model based on validation accuracy
    best_model, best_params, val_accuracies = train_and_select_model(x_train, y_train, x_val, y_val, kernels, C_values)
    # Evaluate the best model on the test set and generate the confusion matrix
    conf_matrix = evaluate_model(best_model, x_test, y_test)

    # Plot validation accuracy for each kernel and its corresponding C values
    plot_validation_accuracy(kernels, C_values, val_accuracies)
    # Plot the confusion matrix of the best SVM model
    plot_confusion_matrix(conf_matrix, best_params)

    # Return the best model, hyperparameters, validation accuracies, and confusion matrix
    return best_model, best_params, val_accuracies, conf_matrix