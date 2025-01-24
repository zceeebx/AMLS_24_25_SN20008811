# Machine Learning Project: BreastMNIST & BloodMNIST Classification

## Project Overview
This project classifies medical images from the BreastMNIST and BloodMNIST datasets using various machine learning models. The implemented models include Logistic Regression, SVM, Decision Tree, and Random Forest. The project structure is modular, allowing for easy management and extension.

## Project Structure
# ├── AMLS_24_25_SN20008811
# │├── A 							# Folder containing Logistic Regression and SVM implementations 
# ││├── pycache 					# Cache files for Python modules 
# ││├── Logistic_Regression.py 	# Logistic Regression implementation 
# ││└── SVM.py 					# Support Vector Machine implementation 
# │├── B 							# Folder containing Decision Tree and Random Forest implementations 
# ││├── pycache 					# Cache files for Python modules 
# ││├── Decision_Tree.py 			# Decision Tree implementation 
# ││└── Random_Forest.py 		# Random Forest implementation 
# │├── Datasets 					# Folder to store BreastMNIST and BloodMNIST datasets (empty by default. place datasets here manually)
# ││└── README.txt				# File instruction
# │├── main.py 					# Main script to execute the classification tasks 
# │├── README.txt 				# Project documentation (this file)


## File Roles
- **`A/`**: Contains machine learning model implementations for:
  - **`Logistic_Regression.py`**: Logistic Regression model.
  - **`SVM.py`**: Support Vector Machine model.
- **`B/`**: Contains machine learning model implementations for:
  - **`Decision_Tree.py`**: Decision Tree model.
  - **`Random_Forest.py`**: Random Forest model.
- **`Datasets/`**: Directory to store BreastMNIST and BloodMNIST datasets. **Note**: Datasets must be added manually.
- **`main.py`**: The main script to orchestrate and run classification tasks using models from folders `A/` and `B/`.
- **`README.md`**: Provides an overview of the project, its structure, and setup instructions.

## Required Libraries
The following external libraries are required to implement this project:
- **`numpy`**: Used for numerical operations and array handling.
- **`matplotlib`**: Used for visualizing model performance, such as accuracy plots and learning curves.
- **`seaborn`**: Provides a high-level interface for creating attractive and informative statistical graphics.
- **`itertools`**: Used to generate the Cartesian product of input iterables.
- **`sklearn`**: Provides tools for data analysis, model building, and evaluation, including:
  - **`MinMaxScaler`**: Scales features to a specified range.
  - **`PCA`**: Reduces data dimensionality.
  - **`LDA`**: Reduces dimensionality while maximizing class separability.
  - **`accuracy_score`**: Calculates model accuracy.
  - **`confusion_matrix`**: Computes the confusion matrix.
  - **`classification_report`**: Generates a report with precision, recall, F1-score, and accuracy.
  - **`ConfusionMatrixDisplay`**: Visualizes confusion matrices.
  - **`KFold`**: Splits datasets for K-Fold cross-validation.
  - **`SVC`**: Implements Support Vector Machines for classification.
  - **`DecisionTreeClassifier`**: Builds a decision tree model.
  - **`RandomForestClassifier`**: Builds a random forest model.
