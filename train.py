import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# 1. Load the MNIST data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 2. Reshape for a CNN (Add the "1" for the grayscale color channel)
# We also divide by 255 to normalize the pixel values between 0 and 1
X_train = X_train.reshape(-1, 28, 28, 1) / 255.0
X_test = X_test.reshape(-1, 28, 28, 1) / 255.0

# Convert labels to One-Hot Encoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# 3. Build the True CNN Architecture
print("Building CNN Model...")
model = Sequential([
    # The Convolutional Base (Feature Extraction)
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    # The Classification Head
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 4. Train the model (5 epochs is usually enough for ~98% accuracy on MNIST)
print("Training CNN... This may take a minute or two.")
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, batch_size=200)

# 5. Save the upgraded model
model.save('handwritten_model.keras')
print("CNN Model saved successfully as handwritten_model.keras!")