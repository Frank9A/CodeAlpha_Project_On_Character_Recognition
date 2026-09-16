import tensorflow as tf
import matplotlib.pyplot as plt

#load the dataset of handwritten numbers (0-9)
mnist = tf.keras.datasets.mnist

#Unpack the dataset int training and testing sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#lets actually look at the very first image!
plt.imshow(x_train[0], cmap='gray')
plt.show()

# Scale the pixel values to be between 0 and 1
x_train = x_train /255.0
x_test = x_test / 255.0

# Build the layers of the Neural Network
model = tf.keras.models.Sequential([
    #Layer 1: Flatten the 28x28 image grid into a single line of pixels
    tf.keras.layers.Flatten(input_shape=(28, 28)),

    # Layer 2: The "hidden" thinking layer where patterns are learned
    tf.keras.layers.Dense(128, activation='relu'),

    # Layer 3: The output layer with 10 options (numbers 0 trough 9)
    tf.keras.layers.Dense(10, activation='softmax')
])

#Give the brain instructions on how to learn
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

#Train the model (the actual learning process)
print("Starting Training...")
model.fit(x_train, y_train, epochs=5)

#Quiz the model on the unseen test data
print("\nEvaluatingmodel on test data...")
test_loss, test_acc = model.evaluate(x_test,y_test, verbose=2)
print(f"\nFinal test accuracy: {test_acc * 100:2f}%")

# Save the trained brain to a file so we can use it in our web API later
model.save('handwritten_model.keras')
print("Model saved successfully!")