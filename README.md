# **SVR Model for Predicting Principal Stress in Cortical Threads**

This repository contains MATLAB code for building, training, and evaluating a Support Vector Regression (SVR) model to predict the principal stress in cortical threads based on key design parameters like thread pitch, depth, and angles. The project includes several scripts for preprocessing data, performing hyperparameter tuning, visualizing results, and simplifying the SVR model by using top support vectors.

## **Table of Contents**
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Training the SVR Model](#training-the-svr-model)
  - [Evaluating the SVR Model](#evaluating-the-svr-model)
- [Repository Structure](#repository-structure)
- [Contributing](#contributing)
- [License](#license)

## **Overview**

The SVR model is designed to predict the **principal stress** in cortical thread designs. It uses five key features of thread geometry:
1. Thread pitch
2. Thread depth
3. Alpha1 (first thread angle)
4. Alpha2 (second thread angle)
5. Thread width

The code allows for outlier detection, scaling of features, hyperparameter tuning, and model evaluation using metrics such as Mean Squared Error (MSE) and R-squared score.

## **Features**

- **Data Preprocessing:** Includes Z-score-based outlier detection and data scaling.
- **SVR Model Training:** Uses grid search for hyperparameter tuning (Box Constraint `C` and Epsilon).
- **Visualization Tools:** Generates plots for correlation matrices, residuals, and actual vs predicted values.
- **Simplified SVR Model:** Option to generate a reduced SVR equation using the top `k` support vectors.
- **Model Evaluation:** Calculates performance metrics on both training and testing datasets.

## **Installation**

To use this project, follow the steps below:

1. Clone the repository to your local machine:

    ```bash
    git clone https://github.com/your-username/SVR_Model_Project.git
    ```

2. Ensure that you have MATLAB installed on your system. No additional toolboxes are required for the basic SVR functionality.

## **Usage**

### **Training the SVR Model**

1. Place your data file in the working directory or provide the path to the data file. The data should be in a CSV format with the following feature columns:
   - `Thread pitch`
   - `Thread depth`
   - `Alpha1`
   - `Alpha2`
   - `Thread width`
   - Target column (e.g., `principal stress-cortical`)

2. Run the `SVRML.m` script to train the SVR model:

    ```matlab
    SVRML('your-data-file.csv');
    ```

This script will:
- Load the data and preprocess it (e.g., remove outliers, scale features).
- Perform hyperparameter tuning using grid search to find the best SVR model.
- Save the trained model and visualizations (e.g., residuals, correlation matrices) in an output directory.

### **Evaluating the SVR Model**

To evaluate a saved SVR model on new data, use the `evaluateFullSVRModel.m` script:

1. Set the path to your saved model and new data file.
2. Set the number of top support vectors to use in the simplified model (e.g., `k = 20`).

Run the script:

```matlab
evaluateFullSVRModel('path/to/saved_model.mat', 'new-data-file.csv', featureColumns, targetColumn, k);
```

This script will:
- Load the saved model and scaling parameters.
- Evaluate the model's performance on the new data.
- Optionally, create a simplified SVR model using the top `k` support vectors.

## **Repository Structure**

- `SVRML.m`: Main script for training the SVR model, tuning hyperparameters, and generating predictions.
- `helper_functions.m`: Contains helper functions for data preprocessing, plotting, and cross-validation.
- `evaluateFullSVRModel.m`: Script for evaluating the performance of a saved SVR model on new data.
- `data/`: Placeholder directory for datasets.
- `output/`: Contains trained models and generated plots.

## **Contributing**

Contributions are welcome! If you'd like to improve the code or add new features, feel free to fork the repository and create a pull request. Please ensure your code is well-documented and tested.

## **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
