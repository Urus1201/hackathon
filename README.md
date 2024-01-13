# Data Science Global Hackathon 2023-24

The aim of the Hackathon is to perform `Zero-shot-Learning` where labeled data along with some side information are available. The model needs to learn this realtionship such that given new features, the model should be able to predict the text embeddings.

## Table of Contents

- [Project Name](#project-name)
  - [Table of Contents](#table-of-contents)
  - [Description](#description)
  - [Features](#features)
  - [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Usage](#installation)
  - [Author](#author)
  - [License](#license)


## Description

- **Code**: The `code` folder comprises all the necessary scripts for either (a) training or (b) making predictions. Please refer to the Getting Started section for detailed instructions.

- **Data**: The `data` folder houses the entire dataset. 
  - The `train` 📂 directory contains files from train set 1.
  - The `test` 📂 directory includes files from test set 2.
  - The `train_test` 📂 directory consolidates files from both train set 1 and test set 1, designed for model training.

- **Model**: In the `model` folder, you'll find the saved model obtained after the training process.

- **Notebooks**: The `notebooks` folder holds notebooks used during the training process. These notebooks are optimized for execution on Google Colab. Be sure to adjust the file paths if running on a different environment.

- **Results**: The `results` folder contains model predictions submitted on Kaggle.

📝 Note: If utilizing the script to generate predictions, ensure a thorough review of folder paths in the `path_config.json` file.


## Features

Besides the given simple MLP following changes has been implemented to enhance model performance:

- **Class Balancing**: To address class imbalance within the initial training set, the **RandomOverSampler** technique is applied in the `DATA_LOADER_HK` class. This ensures a more balanced representation of minority classes in the training data.

- **Attribute Normalization**: Prior to being fed into the model, the attributes (side information) undergo normalization, bringing their values within the range of 0 to 1. This preprocessing step contributes to improved model convergence and robustness.

- **Model Architecture**: The model incorporates a Multi-head Cross Attention mechanism, allowing it to learn intricate relationships between features and attributes. This is particularly beneficial for capturing nuanced dependencies and enhancing the overall performance of the Dense Neural Network.

These enhancements collectively contribute to a more robust and effective model for your project.

## Getting Started

### Prerequisites

To run the project install the dependencies from `requirements.txt`. By 

```python
# Install required dependencies
pip install -r requirements.txt
```

### Usage

What do you want to do?

**A. Train Model:** Navigate to `code` folder in your terminal and then type in the following command : 
```python
# Re-train the model
python train.py
```

**B. Predict Results:** Navigate to `code` folder in your terminal and then type in the following command :
```python
# Predict the semantic attributes
python predict.py
```
📝**Note:** The final prediction result is stored in `results` folder. You can upload it to kaggle from there.

## Author

Sourav Mukherjee

## Lincense

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

MIT License