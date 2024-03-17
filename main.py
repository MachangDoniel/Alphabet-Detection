import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk
import os

# Function to preprocess the input image
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 150)
    return edges

# Function to precompute and threshold template images
def preprocess_templates(template_dir):
    preprocessed_templates = {}
    for alphabet_file in os.listdir(template_dir):
        template_path = os.path.join(template_dir, alphabet_file)
        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        _, thresh = cv2.threshold(template, 1, 255, cv2.THRESH_BINARY)
        preprocessed_templates[alphabet_file] = thresh
    return preprocessed_templates

# Function to compare images based on pixel-wise similarity
def compare_images(input_processed, template, template_thresh):
    # Calculate Mean Squared Error (MSE) only for non-zero pixels
    mse = np.mean((input_processed - template_thresh) ** 2)

    # Calculate percentage similarity only for non-zero pixels
    non_zero_pixels = np.sum(template_thresh == 255)
    similarity_percentage = (np.sum(template_thresh == input_processed) / non_zero_pixels) * 100
    
    return mse, similarity_percentage

# Function to process and compare input image with template images
def recognize_alphabet(input_image, preprocessed_templates):
    input_processed = preprocess_image(input_image)
    min_mse = float('inf')
    max_similarity_percentage = 0
    recognized_alphabet = None

    for alphabet_file, template_thresh in preprocessed_templates.items():
        template_mse, similarity_percentage = compare_images(input_processed, preprocessed_templates[alphabet_file], template_thresh)

        # Print MSE and corresponding alphabet being compared
        print(f"Comparing with alphabet '{os.path.splitext(alphabet_file)[0]}': MSE={template_mse}")

        if template_mse < min_mse:
            min_mse = template_mse
            recognized_alphabet = os.path.splitext(alphabet_file)[0]  # Extract the alphabet from the filename

        if similarity_percentage > max_similarity_percentage:
            max_similarity_percentage = similarity_percentage

        # Early stopping if exact match found
        if min_mse == 0 and max_similarity_percentage == 100:
            break

    # Print recognition results
    print("Recognized Alphabet:", recognized_alphabet)
    print("Minimum MSE:", min_mse)
    print("Max Similarity Percentage:", max_similarity_percentage)

    return recognized_alphabet, min_mse, max_similarity_percentage

# Function to open a file dialog and get the path to the selected image
def select_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    if file_path:
        input_image_path.set(file_path)
        load_image(file_path)

# Function to load and display the selected image
def load_image(image_path):
    image = cv2.imread(image_path)
    # Convert image from BGR to RGB format
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Resize image to fit the UI (optional)
    image = cv2.resize(image, (300, 300))
    # Convert image to PhotoImage format
    photo = ImageTk.PhotoImage(image=Image.fromarray(image))
    # Update the label to display the selected image
    image_label.config(image=photo)
    image_label.image = photo

# Function to perform alphabet detection
def detect_alphabet():
    input_path = input_image_path.get()
    if input_path:
        # Preprocess templates
        preprocessed_templates = preprocess_templates(template_dir)

        # Perform alphabet detection
        recognized_alphabet, min_mse, max_similarity_percentage = recognize_alphabet(cv2.imread(input_path), preprocessed_templates)
        
        # Update result labels
        result_label.config(text=f"Recognized Alphabet: {recognized_alphabet}")
        mse_label.config(text=f"Minimum MSE: {min_mse}")
        similarity_label.config(text=f"Max Similarity Percentage: {max_similarity_percentage}")
    else:
        print("Please select an input image.")

# Create the main application window
root = tk.Tk()
root.title("Alphabet Detection")

# Create a frame for the title
title_frame = tk.Frame(root)
title_frame.pack(pady=10)

# Add a label for the title
title_label = tk.Label(title_frame, text="Alphabet Detection", font=("Arial", 18))
title_label.pack()

# Create a frame for the input image selection
input_frame = tk.Frame(root)
input_frame.pack(pady=10)

# Add a label and button for selecting the input image
input_label = tk.Label(input_frame, text="Select Input Image:")
input_label.grid(row=0, column=0)

input_image_path = tk.StringVar()
input_entry = tk.Entry(input_frame, textvariable=input_image_path, width=40)
input_entry.grid(row=0, column=1)

browse_button = tk.Button(input_frame, text="Browse", command=select_image)
browse_button.grid(row=0, column=2)

# Create a frame for displaying the selected image
image_frame = tk.Frame(root)
image_frame.pack(pady=10)

# Add a label for displaying the selected image
image_label = tk.Label(image_frame)
image_label.pack()

# Create a frame for displaying the results
result_frame = tk.Frame(root)
result_frame.pack(pady=10)

# Add labels for displaying the results
result_label = tk.Label(result_frame, text="Recognized Alphabet:")
result_label.pack()

mse_label = tk.Label(result_frame, text="Minimum MSE:")
mse_label.pack()

similarity_label = tk.Label(result_frame, text="Max Similarity Percentage:")
similarity_label.pack()

# Add a button to perform alphabet detection
detect_button = tk.Button(root, text="Detect Alphabet", command=detect_alphabet)
detect_button.pack(pady=10)

# Directory containing template alphabet images
template_dir = 'temp_edges'  # Adjust this directory as needed

# Start the main event loop
root.mainloop()
