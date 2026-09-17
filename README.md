# Handwritten Character Recognition 

A full-stack machine learning application that predicts handwritten digits (0-9) in real-time. This project features a custom drawing canvas built with React and a Convolutional Neural Network (CNN) served via a Python backend.

This project was built as part of the CodeAlpha Machine Learning Internship.

## Features
* **Interactive Canvas:** An HTML5 drawing surface built with React that captures user input.
* **Smart Preprocessing:** Uses SciPy to calculate the "Center of Mass" of the user's drawing, dynamically centering the raw pixel data to match the MNIST dataset's original lab conditions.
* **Deep Learning AI:** A TensorFlow/Keras Neural Network trained on the MNIST dataset, achieving ~98% test accuracy.

## Tech Stack
* **Frontend:** React, Vite, JavaScript, HTML5 Canvas
* **Backend:** Python, FastAPI, Uvicorn
* **Machine Learning:** TensorFlow, Keras, NumPy, SciPy

## Model Architecture
* Flatten Layer (28x28 to 784 pixels)
* Dense Hidden Layer (128 neurons, ReLU activation)
* Dense Output Layer (10 neurons, Softmax activation)
* CNN with Conv2D and MaxPooling2D