from itertools import product
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

def train_and_validate(X_train_flat, y_train, X_val_flat, y_val):
    """
    Tunes hyperparameters for a Random Forest model using cross-validation, and returns the best model.

    Parameters:
    - X_train_flat (ndarray): The flattened training dataset features.
    - y_train (ndarray): The training dataset labels.
    - X_val_flat (ndarray): The flattened validation dataset features.
    - y_val (ndarray): The validation dataset labels.

    Returns:
    - best_rf_model (RandomForestClassifier): The trained Random Forest model with the best hyperparameters.
    """
    # Set the hyperparameter grid for Random Forest
    rf_param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': [None, 'sqrt']
    }

    # Generate all possible combinations of hyperparameters
    rf_param_combinations = list(product(*rf_param_grid.values()))

    best_rf_accuracy = 0
    best_rf_params = None
    best_rf_model = None

    # Store results for all hyperparameter combinations and their validation accuracies
    results = []

    # Loop through all hyperparameter combinations to train and evaluate the model
    for params in rf_param_combinations:
        # Package the current combination of parameters into a dictionary
        param_dict = dict(zip(rf_param_grid.keys(), params))

        # Initialize the Random Forest model with the current set of parameters
        rf_model = RandomForestClassifier(random_state=42, **param_dict)
        rf_model.fit(X_train_flat, y_train)

        # Evaluate the model on the validation set
        y_val_pred = rf_model.predict(X_val_flat)
        val_accuracy = accuracy_score(y_val, y_val_pred)

        # Print the current hyperparameter combination and its validation accuracy
        print(f"Hyper-parameter combination: {param_dict}, Validation accuracy: {val_accuracy:.4f}")
        
        # Save the results of this hyperparameter combination
        results.append({'params': param_dict, 'val_accuracy': val_accuracy})
        
        # Track the best performing model and its parameters
        if val_accuracy > best_rf_accuracy:
            best_rf_accuracy = val_accuracy
            best_rf_params = param_dict
            best_rf_model = rf_model

    # Print the best hyperparameter combination and its validation accuracy
    print("Best hyper-parameter combination:", best_rf_params)
    print(f'Validation accuracy: {best_rf_accuracy * 100:.2f}%')

    # Visualize the results and mark the highest point (best accuracy)
    visualize_results(results, best_rf_accuracy)

    # Return the best performing Random Forest model
    return best_rf_model

def visualize_results(results, best_accuracy):
    """
    Visualizes the results of hyperparameter tuning for a Random Forest model using a bar chart.
    Marks the best hyperparameter combination with the highest validation accuracy.

    Parameters:
    - results (list): A list of dictionaries containing hyperparameter combinations and their corresponding validation accuracies.
    - best_accuracy (float): The highest validation accuracy achieved during the hyperparameter tuning process.

    Returns:
    - None: This function only visualizes the results and displays the plot.
    """
    # Extract the validation accuracies from the results list
    val_accuracies = [r['val_accuracy'] for r in results]

    # Find the index of the best validation accuracy
    best_index = val_accuracies.index(best_accuracy)

    # Create a bar chart to visualize the validation accuracies for all hyperparameter combinations
    plt.figure(figsize=(15, 7))                             # Set the figure size for better visibility
    x = range(len(results))                                 # x-axis represents the different hyperparameter combinations
    plt.bar(x, val_accuracies, color='skyblue', alpha=0.8)  # Create bars for validation accuracies
    
    # Mark the best validation accuracy with a red point
    plt.scatter(best_index, best_accuracy, color='red', s=100, label=f"Best ({best_accuracy:.4f})")
    plt.annotate(f"{best_accuracy:.4f}", 
                 (best_index, best_accuracy),  # Position of the annotation
                 textcoords="offset points",   # Offset the annotation text slightly
                 xytext=(-10, 10),             # Adjust the position of the annotation text
                 ha='center',                  # Horizontal alignment of the annotation
                 fontsize=10,                  # Font size of the annotation
                 color='red')                  # Color of the annotation text

    # Set the labels and title for the plot
    plt.xlabel('Hyper-parameter Combinations')  # Label for the x-axis
    plt.ylabel('Validation Accuracy')           # Label for the y-axis
    plt.title('Random Forest Hyper-parameter Tuning Results')  # Title of the plot
    plt.legend()                                # Display the legend to show what the red point represents
    plt.tight_layout()                          # Adjust layout to prevent overlap
    plt.show()                                  # Display the plot

def y_predict(best_rf_model, X_test_flat, y_test):
    """
    Evaluates the best Random Forest model on the test set and visualizes the confusion matrix.

    Parameters:
    - best_rf_model (RandomForestClassifier): The trained Random Forest model with optimal hyperparameters.
    - X_test_flat (ndarray): The flattened feature set of the test dataset.
    - y_test (ndarray): The true labels for the test dataset.

    Returns:
    - tuple: The predicted labels for the test set (`y_test_pred`) and the confusion matrix (`val_cm`).
    """
    # Make predictions on the test set using the trained Random Forest model
    y_test_pred = best_rf_model.predict(X_test_flat)

    # Print the confusion matrix to evaluate the model's performance
    print(confusion_matrix(y_test, y_test_pred))
    # Print the accuracy score of the model on the test set
    print(f"Accuracy on test set: {accuracy_score(y_test, y_test_pred):.4f}")
    # Print a detailed classification report, which includes precision, recall, F1 score, and support for each class
    print(classification_report(y_test, y_test_pred, zero_division=1))

     # Display the confusion matrix visually
    val_cm = confusion_matrix(y_test, y_test_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=val_cm)
    disp.plot(cmap=plt.cm.Blues)                        # Plot using a blue color map for better visualization
    plt.title('Best Random Forest Confusion Matrix')    # Title for the plot
    plt.show()                                          # Show the plot

    # Return the predicted labels and confusion matrix for further analysis
    return y_test_pred, val_cm

def random_forest(X_train_flat, y_train, X_val_flat, y_val, X_test_flat, y_test):
    """
    Trains a Random Forest model, performs hyperparameter tuning, evaluates the model, and returns results.

    Parameters:
    - X_train_flat (ndarray): Flattened feature set of the training dataset.
    - y_train (ndarray): Labels of the training dataset.
    - X_val_flat (ndarray): Flattened feature set of the validation dataset.
    - y_val (ndarray): Labels of the validation dataset.
    - X_test_flat (ndarray): Flattened feature set of the test dataset.
    - y_test (ndarray): Labels of the test dataset.

    Returns:
    - tuple: The trained Random Forest model (`best_rf_model`), the predicted labels for the test set (`y_test_pred`),
             and the confusion matrix (`val_cm`).
    """
    # Step 1: Train and tune the Random Forest model using hyperparameter grid search
    best_rf_model = train_and_validate(X_train_flat, y_train, X_val_flat, y_val)
    
    # Step 2: Evaluate the model on the test set
    y_test_pred, val_cm = y_predict(best_rf_model, X_test_flat, y_test)

    # Step 3: Return the trained model, predicted labels, and confusion matrix
    return best_rf_model, y_test_pred, val_cm
