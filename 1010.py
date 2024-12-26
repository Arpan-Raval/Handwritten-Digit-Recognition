import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from PIL import Image, ImageDraw
import tkinter as tk

# Load the MNIST dataset
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist.data.values, mnist.target.astype(int)

# Normalize pixel values to the range [0, 1]
X = X / 255.0

# Display some sample images
plt.figure(figsize=(10, 5))
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(X[i].reshape(28, 28), cmap='gray')
    plt.title(f"Label: {y[i]}")
    plt.axis('off')
plt.show()

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the MLP Classifier
clf = MLPClassifier(hidden_layer_sizes=(256, 128, 64), max_iter=240, early_stopping=True, 
                    validation_fraction=0.1, alpha=1e-4, solver='adam', verbose=True, 
                    random_state=1, learning_rate_init=0.001)

# Train the model
clf.fit(X_train, y_train)

# Print training accuracy
train_accuracy = clf.score(X_train, y_train)
print(f"Training Accuracy: {train_accuracy:.2f}")

# Test the model
y_pred = clf.predict(X_test)

# Calculate accuracy
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {test_accuracy:.2f}")

# Classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()


# Function to preprocess and predict digit from a drawn image
def predict_digit(image_path):
    # Load and preprocess the image
    img = Image.open(image_path).convert("L")  # Convert to grayscale
    img_array = np.array(img).reshape(1, -1)  # Flatten and reshape
    img_array = 255 - img_array  # Invert colors to match MNIST
    img_array = img_array / 255.0  # Normalize pixel values

    # Predict the digit
    prediction = clf.predict(img_array)
    print(f"Predicted Digit: {prediction[0]}")

    # Display the image and prediction
    plt.imshow(img, cmap="gray")
    plt.title(f"Predicted Digit: {prediction[0]}")
    plt.axis("off")
    plt.show()


# Function to create a drawing canvas
def draw_digit():
    # Create a new window for the canvas
    root = tk.Tk()
    root.title("Draw a Digit")
    canvas_width, canvas_height = 280, 280

    # Set up a canvas for drawing
    canvas = tk.Canvas(root, width=canvas_width, height=canvas_height, bg="white")
    canvas.pack()

    # Set up a PIL image to draw on
    drawing_image = Image.new("L", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(drawing_image)

    # Function to draw on the canvas
    def paint(event):
        x1, y1 = (event.x - 5), (event.y - 5)
        x2, y2 = (event.x + 5), (event.y + 5)
        canvas.create_oval(x1, y1, x2, y2, fill="black", outline="black")
        draw.ellipse([x1, y1, x2, y2], fill="black")

    canvas.bind("<B1-Motion>", paint)

    # Function to save the drawing and predict the digit
    def save_drawing():
        # Save the drawn image
        drawing_image_resized = drawing_image.resize((28, 28))
        drawing_image_resized.save("drawing.png")

        # Predict the digit
        predict_digit("drawing.png")

        root.destroy()

    # Add a button to save the drawing
    save_button = tk.Button(root, text="Predict", command=save_drawing)
    save_button.pack()

    root.mainloop()


# Menu for the application
def menu():
    while True:
        print("\nMenu:")
        print("1. Draw a digit")
        print("2. Exit")
        try:
            choice = int(input("Enter your choice: "))
            if choice == 1:
                draw_digit()
            elif choice == 2:
                print("Exiting. Goodbye!")
                break
            else:
                print("Invalid choice. Please select 1 or 2.")
        except ValueError:
            print("Invalid input. Please enter a number.")

# Run the menu
menu()