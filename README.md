# Sign Language MNIST Classifier and Real-Time Detection

## Overview

This repository contains code for training a Sign Language MNIST classifier using Convolutional Neural Networks (CNN) and performing real-time sign language detection using a webcam. The model is built with TensorFlow and Keras.

## Installation

Install the required dependencies by running:

```bash
pip install -r requirements.txt
```
## Usage
# Training the Model
1.Download the Sign Language MNIST dataset from https://www.kaggle.com/datasets/datamunge/sign-language-mnist.

2.Extract the dataset:

```bash
from zipfile import ZipFile

with ZipFile('path/to/archive.zip', 'r') as zip_ref:
    zip_ref.extractall('path/to/dataset')
```

3.Train the model:

```bash
python train_model.py
Evaluate the model:
```

## Real-Time Detection

1.Ensure the trained model is saved as 'sign_language_cnn_model.keras'.

2.Run real-time detection:

```bash
python real_time_detection.py
```

## Files and Structure

- train_model.py: Script to train the CNN model.
- evaluate_model.py: Script to evaluate the model on the test set.
- real_time_detection.py: Real-time sign language detection script.
- requirements.txt: List of required Python packages.

## Dataset 

![dataset](https://github.com/paramsureliya/Sign-Language-MNIST-Kaggle/assets/148708744/3d5a2ad4-4c87-45d0-abfb-b5876bf54281)

## Contributing
Contributions are welcome! Feel free to open issues or pull requests.
