# MaterialShellClassifier


# Material Shell Classifier

A PyTorch-based 1D Convolutional Neural Network model that classifies hollow material shells into three categories:

* Weak Shell
* Moderate Shell
* Optimal Shell

The model uses numerical material/property parameters from a CSV dataset and predicts the shell quality class.

## Project Overview

This project trains a machine learning model to classify hollow shell structures based on input material parameters. The workflow includes:

1. Loading and cleaning the dataset
2. Mapping text labels into numerical classes
3. Scaling input features
4. Building a custom PyTorch Dataset and DataLoader
5. Training a 1D CNN classifier
6. Evaluating model accuracy
7. Saving the trained model and scaler
8. Running predictions on new shell parameter inputs

## Dataset

The model uses a dataset named:

```text
HollowShell_Dataset.csv
```

The dataset contains numerical input features and a final label column representing the shell category.

The label mapping is:

```python
{
    "Weak_Shell": 0,
    "Moderate_Shell": 1,
    "Optimal_Shell": 2
}
```

## Model Architecture

The classifier is built using PyTorch and contains:

* 1D convolutional layer
* ReLU activation
* Second 1D convolutional layer
* Max pooling
* Adaptive average pooling
* Flatten layer
* Fully connected dense layers
* Final output layer with 3 classes

The model uses `CrossEntropyLoss` because this is a multi-class classification problem.

## Training Setup

The dataset is split into:

* 80% training data
* 20% testing data

The model is trained using:

```python
optimizer = Adam
learning_rate = 0.01
epochs = 10
batch_size = 64
loss_function = CrossEntropyLoss
```

## Results

During training, the model reached strong validation performance, with validation accuracy reaching about:

```text
95% - 96%
```

Example final validation output:

```text
Validation Loss: 0.11, Accuracy: 0.95
```

## Saved Files

After training, the project saves:

```text
MaterialModel.pth
```

This contains the trained PyTorch model weights.

```text
materialScaler
```

This contains the fitted StandardScaler used to scale future input data.

## Example Prediction

Example input parameters:

```python
input_params = [
    123.24, 1021.32, 0.682, 1.340, 28.345,
    342.34, 2.9342, 8.234, 1.213, 3.324,
    9.642, 0.783, 0.7832
]
```

Model output:

```text
tensor([1])
```

Final prediction:

```text
According to your parameters the predicted shell is a Moderate Shell type.
```

## Important Note

When using the model for future predictions, the same scaler fitted on the training data should be reused. Do not fit a new scaler on a single prediction input, because that can distort the data.

Use the saved scaler:

```python
input_scaler = joblib.load("materialScaler")
input_scaled = input_scaler.transform(input_params)
```

## Future Improvements

Possible improvements include:

* Add a confusion matrix
* Save class names in a separate file
* Add a prediction script
* Add model evaluation metrics like precision, recall, and F1-score
* Improve training loss averaging
* Add GPU support
* Package the model into a simple app or API

## Technologies Used

* Python
* PyTorch
* Pandas
* NumPy
* Scikit-learn
* Joblib

## Author

Created by Veltman Okey-Ejiowhor as a machine learning project for material shell classification.
