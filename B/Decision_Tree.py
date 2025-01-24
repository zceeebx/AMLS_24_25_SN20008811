from itertools import product
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

def train_and_validate(X_train_flat, y_train, X_val_flat, y_val):
    """
    Trains and validates a Decision Tree model using hyperparameter tuning via grid search.

    Steps:
    1. Defines a grid of hyperparameters for the Decision Tree model.
    2. Evaluates all combinations of hyperparameters using the training and validation datasets.
    3. Tracks and returns the best performing model based on validation accuracy.

    Parameters:
    - X_train_flat (ndarray): Flattened training dataset features.
    - y_train (ndarray): Training dataset labels.
    - X_val_flat (ndarray): Flattened validation dataset features.
    - y_val (ndarray): Validation dataset labels.

    Returns:
    - best_dt_model (DecisionTreeClassifier): The best Decision Tree model based on validation accuracy.
    """
    # Define a grid of hyperparameters for tuning the Decision Tree model
    dt_param_grid = {
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4, 6],
        'max_features': [None, 'sqrt', 'log2']
    }

    # Generate all possible combinations of hyperparameters from the grid
    dt_param_combinations = list(product(*dt_param_grid.values()))

    best_dt_accuracy = 0
    best_dt_params = None
    best_dt_model = None

    # List to store results for each combination of hyperparameters
    results = []

    # Iterate through all hyperparameter combinations and evaluate them on the validation set
    for params in dt_param_combinations:
        # Convert the combination into a dictionary of hyperparameters
        param_dict = dict(zip(dt_param_grid.keys(), params))
        
        # Initialize and train the Decision Tree model using the current combination of hyperparameters        
        dt_model = DecisionTreeClassifier(random_state=42, **param_dict)
        dt_model.fit(X_train_flat, y_train)

        # Evaluate the model on the validation set and compute the accuracy        
        y_val_pred = dt_model.predict(X_val_flat)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        print(f"hyper-parameter combination: {param_dict}, validation accuracy: {val_accuracy:.4f}")
        
        # Store the results
        results.append({'params': param_dict, 'val_accuracy': val_accuracy})
        
        # Track the best performing model based on validation accuracy
        if val_accuracy > best_dt_accuracy:
            best_dt_accuracy = val_accuracy
            best_dt_params = param_dict
            best_dt_model = dt_model

    # Output the best hyperparameter combination and its corresponding accuracy
    print("optimal hyper-parameter combination:", best_dt_params)
    print(f'validation accuracy: {best_dt_accuracy:.2f}')

    # Visualize the results, highlighting the best performing hyperparameter combination
    visualize_results(results, best_dt_accuracy)

    # Return the best performing Decision Tree model
    return best_dt_model

def visualize_results(results, best_accuracy):
    """
    Visualizes the hyperparameter tuning results by plotting a bar chart showing the validation accuracy
    for each hyperparameter combination and highlighting the best performing combination.

    Parameters:
    - results (list): List of dictionaries containing hyperparameter combinations and their corresponding
                      validation accuracies.
    - best_accuracy (float): The best validation accuracy achieved by any hyperparameter combination.
    """
    # Extract the hyperparameter combinations and their validation accuracies
    labels = [str(r['params']) for r in results]
    val_accuracies = [r['val_accuracy'] for r in results]

    # Find the index of the best performing hyperparameter combination
    best_index = val_accuracies.index(best_accuracy)

    # Create the bar chart
    plt.figure(figsize=(15, 7)) # Set the figure size
    x = range(len(results))     # X-axis positions for each bar (corresponding to each hyperparameter combination)
    # Create the bars
    plt.bar(x, val_accuracies, color='skyblue', alpha=0.8)
    # Mark the best performing point
    plt.scatter(best_index, best_accuracy, color='red', s=100, label=f"Best ({best_accuracy:.4f})")
    # Annotate the best point with its value
    plt.annotate(f"{best_accuracy:.4f}", 
                 (best_index, best_accuracy), 
                 textcoords="offset points", 
                 xytext=(-10, 10),  # Position the annotation slightly above and to the left of the point
                 ha='center', 
                 fontsize=10, 
                 color='red')

    # Configure the axis labels and title
    plt.xlabel('Hyper-parameter Combinations')      # X-axis label
    plt.ylabel('Validation Accuracy')               # Y-axis label
    plt.title('Decision Tree Hyper-parameter Tuning Results')   # Plot title
    plt.legend()                                    # Show the legend
    plt.tight_layout()                              # Adjust layout for better spacing
    plt.show()                                      # Display the plot

def y_predict(best_dt_model, X_test_flat, y_test):
    """
    Evaluates the best Decision Tree model on the test dataset and displays performance metrics.

    Parameters:
    - best_dt_model (DecisionTreeClassifier): The trained Decision Tree model with the best hyperparameters.
    - X_test_flat (ndarray): The flattened test dataset features.
    - y_test (ndarray): The actual labels for the test dataset.

    Returns:
    - tuple: Contains two elements:
        - y_test_pred (ndarray): The predicted labels for the test dataset.
        - val_cm (ndarray): The confusion matrix of the predictions.
    """
    # Use the best decision tree model to predict on the test set
    y_test_pred = best_dt_model.predict(X_test_flat)

    # Print the confusion matrix, accuracy score, and classification report
    print(confusion_matrix(y_test, y_test_pred))                                # Show confusion matrix
    print(f"Accuracy on test set: {accuracy_score(y_test, y_test_pred):.4f}")   # Accuracy score
    print(classification_report(y_test, y_test_pred, zero_division=1))          # Detailed classification report

    # Display the confusion matrix for visual analysis
    val_cm = confusion_matrix(y_test, y_test_pred)                              # Calculate confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=val_cm)                      # Create display object for confusion matrix
    disp.plot(cmap=plt.cm.Blues)                                                # Plot the confusion matrix with a blue color map
    plt.title('Best Decision Tree Confusion Matrix')                            # Set title for the plot
    plt.show()  
    
    # Return predicted values and confusion matrix                                                                # Show the plot
    return y_test_pred, val_cm

def decision_tree(X_train_flat, y_train, X_val_flat, y_val, X_test_flat, y_test):
    """
    Trains a Decision Tree model using hyperparameter tuning, evaluates the model, and returns the results.

    Parameters:
    - X_train_flat (ndarray): The flattened training dataset features.
    - y_train (ndarray): The training dataset labels.
    - X_val_flat (ndarray): The flattened validation dataset features.
    - y_val (ndarray): The validation dataset labels.
    - X_test_flat (ndarray): The flattened test dataset features.
    - y_test (ndarray): The test dataset labels.

    Returns:
    - tuple: Contains three elements:
        - best_dt_model (DecisionTreeClassifier): The trained Decision Tree model with the best hyperparameters.
        - y_test_pred (ndarray): The predicted labels for the test dataset.
        - val_cm (ndarray): The confusion matrix of the predictions on the test set.
    """
    # Train the decision tree model with hyperparameter tuning using the training and validation datasets
    best_dt_model = train_and_validate(X_train_flat, y_train, X_val_flat, y_val)

    # Use the best trained decision tree model to make predictions on the test dataset
    y_test_pred, val_cm = y_predict(best_dt_model, X_test_flat, y_test)

    # Return the best model, predictions, and confusion matrix
    return best_dt_model, y_test_pred, val_cm