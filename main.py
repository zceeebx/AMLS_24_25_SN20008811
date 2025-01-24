import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from A.Logistic_Regression import logistic_regression_classifier
from A.SVM import SVM
from B.Decision_Tree import decision_tree
from B.Random_Forest import random_forest

# enter you own data path
breastMNIST = np.load('C:/Users/zceee/Desktop/ELEC0134_Applied_Machine_Learning_Systems/03_Coursework/AMLS_24_25_SN20008811/Datasets/breastmnist.npz')
bloodMNIST = np.load('C:/Users/zceee/Desktop/ELEC0134_Applied_Machine_Learning_Systems/03_Coursework/AMLS_24_25_SN20008811/Datasets/bloodmnist.npz')

def upload_file(file):
    """
    Loads data from a .npz file and returns the training, validation, and test datasets.

    Parameters:
    - file (np.lib.npyio.NpzFile): The .npz file object containing multiple named arrays.

    Returns:
    - tuple: A tuple containing:
        - x_train: Features of the training dataset.
        - y_train: Labels of the training dataset.
        - x_val: Features of the validation dataset.
        - y_val: Labels of the validation dataset.
        - x_test: Features of the test dataset.
        - y_test: Labels of the test dataset.
    """
    # Print the names of the arrays in the file to confirm its structure
    print(file.files)

    # Extract the training dataset
    x_train = file['train_images']
    y_train = file['train_labels']
    # Extract the validation dataset
    x_val = file['val_images']
    y_val = file['val_labels']
    # Extract the testing dataset
    x_test = file['test_images']
    y_test = file['test_labels']

    # Return all extracted datasets
    return x_train, y_train, x_val, y_val, x_test, y_test

def combine_date(x_train, x_val, y_train, y_val):
    """
    Combines training and validation datasets into a single dataset for cross-validation.

    Parameters:
    - x_train (ndarray): Features of the training dataset.
    - x_val (ndarray): Features of the validation dataset.
    - y_train (ndarray): Labels of the training dataset.
    - y_val (ndarray): Labels of the validation dataset.

    Returns:
    - tuple:
        - x_train_full (ndarray): Combined features of training and validation datasets.
        - y_train_full (ndarray): Combined labels of training and validation datasets.
    """
    # Combine the features and labels of the training and validation datasets along the first axis
    x_train_full = np.concatenate((x_train, x_val), axis=0)
    y_train_full = np.concatenate((y_train, y_val), axis=0)

    # Return the combined features and labels
    return x_train_full, y_train_full

def data_reshape(x, y):
    """
    Reshapes the input data, checks for duplicate rows, and counts NaN values.

    Parameters:
    - x (ndarray): Input features, typically a multi-dimensional array.
    - y (ndarray): Input labels, typically a 1D or 2D array.

    Returns:
    - tuple:
        - x (ndarray): Reshaped feature data, where each row is flattened into a single dimension.
        - y (ndarray): Reshaped label data, converted into a 1D array.
    """
    # Reshape the feature array x to a 2D array where each row is flattened
    x = x.reshape(x.shape[0], -1)
    # Reshape the label array y to a 1D array
    y = y.reshape(-1)

    # Check for duplicate rows in x
    # Convert each row of x into a tuple, as tuples are hashable and can be added to a set
    x_tuples = [tuple(row) for row in x]
    # Create a set of unique rows from the tuples
    unique_rows = set(x_tuples)
    # Calculate the number of duplicate rows by comparing the lengths of the original list and the set
    duplicate_count = len(x_tuples) - len(unique_rows)
    print(f"Number of duplicates: {duplicate_count}")
    # Check for NaN values in x and count them
    print(f"Number of NaNs: {np.isnan(x).sum()}")

    # Return the reshaped features and labels
    return x, y

def dualization(y):
    """
    Converts labels in the input array to a binary format by setting values greater than 1 to 0.

    Parameters:
    - y (ndarray): A NumPy array of labels, which may contain integers greater than 1.

    Returns:
    - ndarray: A modified array where all values greater than 1 are set to 0.
    """
    # Update all elements in y that are greater than 1 and set them to 0
    y[y > 1] = 0

    # Return the modified array
    return y

def scale_data(x_train, x_val, x_test, x_train_full=None):
    """
    Scales the input data using MinMaxScaler to normalize values to a specific range (default: [0, 1]).

    Parameters:
    - x_train (ndarray): Training dataset features.
    - x_val (ndarray): Validation dataset features.
    - x_test (ndarray): Test dataset features.
    - x_train_full (ndarray, optional): Combined training and validation dataset for cross-validation (default: None).

    Returns:
    - tuple: Scaled datasets:
        - x_train (ndarray): Scaled training dataset.
        - x_val (ndarray): Scaled validation dataset.
        - x_test (ndarray): Scaled test dataset.
        - x_train_cross (ndarray or None): Scaled combined training dataset (if `x_train_full` is provided).
        - x_test_cross (ndarray or None): Scaled test dataset for cross-validation (if `x_train_full` is provided).
    """
    # Initialize a MinMaxScaler to scale values to the range [0, 1]
    scaler = MinMaxScaler()

    # If x_train_full is provided (e.g., for cross-validation)
    if x_train_full is not None:
        # Fit the scaler to x_train_full and transform it to scale the combined training dataset
        x_train_cross = scaler.fit_transform(x_train_full)
        # Transform x_test using the same scaler to ensure consistent scaling
        x_test_cross = scaler.transform(x_test)

    # Fit the scaler to x_train and transform it
    x_train = scaler.fit_transform(x_train)
    # Transform x_val and x_test using the scaler trained on x_train
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)

    # Return all scaled datasets
    return x_train, x_val, x_test, x_train_cross, x_test_cross

def PCA_data(x_train, x_val, x_test, n_components=0.95):
    """
    Applies Principal Component Analysis (PCA) to reduce the dimensionality of the input data.

    Parameters:
    - x_train (ndarray): Training dataset features.
    - x_val (ndarray): Validation dataset features.
    - x_test (ndarray): Test dataset features.
    - n_components (float or int, optional): Number of principal components to keep.
        - If float (e.g., 0.95), keeps enough components to explain the given variance ratio (default: 0.95).
        - If integer, specifies the exact number of components to retain.

    Returns:
    - tuple: Transformed datasets after applying PCA:
        - x_train_pca (ndarray): Training dataset after PCA transformation.
        - x_val_pca (ndarray): Validation dataset after PCA transformation.
        - x_test_pca (ndarray): Test dataset after PCA transformation.
    """
    # Initialize the PCA object with the specified number of components
    pca = PCA(n_components)

    # Fit the PCA model on the training data and transform it
    x_train_pca = pca.fit_transform(x_train)
    # Transform the validation and testing data using the PCA model fitted on the training data
    x_val_pca = pca.transform(x_val)
    x_test_pca = pca.transform(x_test)

    # Return the transformed datasets
    return x_train_pca, x_val_pca, x_test_pca

def LDA_data(x_train, y_train, x_val, x_test):
    """
    Applies Linear Discriminant Analysis (LDA) for dimensionality reduction on the input datasets.

    Parameters:
    - x_train (ndarray): Training dataset features.
    - y_train (ndarray): Training dataset labels, required for supervised LDA.
    - x_val (ndarray): Validation dataset features.
    - x_test (ndarray): Test dataset features.

    Returns:
    - tuple: Transformed datasets after applying LDA:
        - x_train (ndarray): Training dataset after LDA transformation.
        - x_val (ndarray): Validation dataset after LDA transformation.
        - x_test (ndarray): Test dataset after LDA transformation.
    """
    # Initialize the Linear Discriminant Analysis (LDA) model
    # n_components is set to 7 (choose based on the number of classes in the dataset)
    lda = LDA(n_components=7)

    # Fit LDA on the training data and labels, and transform the training data
    x_train = lda.fit_transform(x_train, y_train)
    # Transform the validation and testing data using the LDA model fitted on the training data
    x_val = lda.transform(x_val)
    x_test = lda.transform(x_test)

    # Return the transformed datasets
    return x_train, x_val, x_test

def taskA():
    """
    Prepares data, applies scaling, and runs machine learning models (Logistic Regression and SVM) on the breastMNIST dataset.

    Steps:
    1. Load the breastMNIST dataset.
    2. Combine training and validation data for cross-validation.
    3. Reshape and dualize the data.
    4. Scale the data.
    5. Train and evaluate Logistic Regression models (both with and without cross-validation).
    6. Apply PCA for dimensionality reduction and run Support Vector Machine (SVM).

    Returns:
    - None: The function prints model results directly.
    """
    # Load the breastMNIST dataset, which contains features (images) and labels for training, validation, and testing.
    x_train, y_train, x_val, y_val, x_test, y_test = upload_file(breastMNIST)
    
    # Combine training and validation datasets for cross-validation. 
    # This combines x_train and x_val, and y_train and y_val into full datasets for later use in cross-validation.
    x_train_full, y_train_full = combine_date (x_train, x_val, y_train, y_val)

    # Reshape and process the data.
    print("\nFor train data:")
    x_train, y_train = data_reshape(x_train, y_train)
    y_train = dualization(y_train)

    print("\nFor validation data:")
    x_val, y_val = data_reshape(x_val, y_val)
    y_val = dualization(y_val)

    print("\nFor test data:")
    x_test, y_test = data_reshape(x_test, y_test)
    y_test = dualization(y_test)

    print("\nFor combined train and validation data:")
    x_train_full, y_train_full = data_reshape(x_train_full, y_train_full)
    y_train_full = dualization(y_train_full)

    # Apply scaling to the datasets to normalize feature values.
    x_train, x_val, x_test, x_train_cross, x_test_cross = scale_data(x_train, x_val, x_test, x_train_full)

    # Run Logistic Regression
    # Train and evaluate Logistic Regression model without cross-validation.
    print("\nRunning Logistic Regression...")
    logistic_model = logistic_regression_classifier(x_train, y_train, x_val, y_val, x_test, y_test, x_train_cross, y_train_full, x_test_cross, k=0)
    # Train and evaluate Logistic Regression model with cross-validation.
    print("\nRunning Logistic Regression with cross-validation...")
    logistic_model = logistic_regression_classifier(x_train, y_train, x_val, y_val, x_test, y_test, x_train_cross, y_train_full, x_test_cross, k=8)

    # Run Support Vector Machine (SVM)
    print("\nRunning Support Vector Machine (SVM) Model...")
    # Apply PCA for dimensionality reduction, then train and evaluate the SVM model.
    x_train, x_val, x_test = PCA_data(x_train, x_val, x_test, n_components=0.95)
    SVM_model = SVM(x_train, y_train, x_val, y_val, x_test, y_test)

def taskB():
    """
    Prepares data and runs machine learning models (Decision Tree and Random Forest) on the bloodMNIST dataset.

    Steps:
    1. Load the bloodMNIST dataset.
    2. Reshape the data for training, validation, and testing.
    3. Apply Linear Discriminant Analysis (LDA) for dimensionality reduction on the data.
    4. Train and evaluate Decision Tree model.
    5. Train and evaluate Random Forest model.

    Returns:
    - None: The function prints the results of the models directly.
    """
    # Load the bloodMNIST dataset, which contains features (images) and labels for training, validation, and testing.
    x_train, y_train, x_val, y_val, x_test, y_test = upload_file(bloodMNIST)

    # Reshape and process the data.
    print("For train data:")
    x_train, y_train = data_reshape(x_train, y_train)

    print("For validation data:")
    x_val, y_val = data_reshape(x_val, y_val)

    print("For test data:")
    x_test, y_test = data_reshape(x_test, y_test)

    # Run Decision Tree
    print("\nRunning Decision Tree...")
    # Apply LDA for dimensionality reduction, then train and evaluate the Decision Tree model.
    x_train_LDA, x_val_LDA, x_test_LDA = LDA_data(x_train, y_train, x_val, x_test)
    decision_tree_results = decision_tree(x_train_LDA, y_train, x_val_LDA, y_val, x_test_LDA, y_test)

    # Run Random Forest
    # Train and evaluate the Random Forest model on the original data.
    print("\nRunning Random Forest...")
    random_forest_results = random_forest(x_train, y_train, x_val, y_val, x_test, y_test)
    
if __name__ == "__main__":
    taskA()
    taskB()
