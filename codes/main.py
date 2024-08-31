import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk
import os

# Segmentation on input image to isolate the alphabets
def segment_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sorting contours wrt their x-coordinate
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

    segments = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        segments.append(image[y:y+h, x:x+w])
    return segments

def preprocess_image(segment):
    resized = cv2.resize(segment, (100, 100))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    return gray

def preprocess_templates(template_dir):
    preprocessed_templates = {}
    for alphabet_file in os.listdir(template_dir):
        template_path = os.path.join(template_dir, alphabet_file)
        template = cv2.imread(template_path)
        if template is not None:
            segments = segment_image(template)
            if segments:
                template_preprocessed = [preprocess_image(segment) for segment in segments]
                preprocessed_templates[alphabet_file] = template_preprocessed
    return preprocessed_templates

def compare_images(input_processed, template):
    
    # cv2.imshow("Input Image",input_processed)
    # cv2.imshow("Template",template)
    # cv2.waitKey(0)

    mse = np.mean((input_processed - template) ** 2)

    non_zero_pixels = np.sum(template != 0)
    
    return mse

def recognize_alphabets(input_image, preprocessed_templates):
    segments = segment_image(input_image)
    recognized_alphabets = []

    for segment in segments:

        cv2.imshow("Segment",segment)
        cv2.waitKey(0)

        input_processed = preprocess_image(segment)
        min_mse = float('inf')
        recognized_alphabet = None

        for alphabet_file, templates in preprocessed_templates.items():
            for template in templates:
                template_mse = compare_images(input_processed, template)

                print(f"Comapring with {alphabet_file[0]}, MSE: {template_mse}")

                if template_mse < min_mse:
                    min_mse = template_mse
                    recognized_alphabet = os.path.splitext(alphabet_file)[0]

                # if min_mse == 0.0:
                #     break
        
        print(f"Alphabet: {recognized_alphabet}, Min MSE: {min_mse}")
        print()
        recognized_alphabets.append((recognized_alphabet, min_mse))

    return recognized_alphabets

def select_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    if file_path:
        input_image_path.set(file_path)
        load_image(file_path)

def load_image(image_path):
    image = cv2.imread(image_path)
    aspect_ratio = image.shape[1] / image.shape[0]
    max_height = 200
    target_width = int(max_height * aspect_ratio)
    image = cv2.resize(image, (target_width, max_height))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    photo = ImageTk.PhotoImage(image=Image.fromarray(image_rgb))
    image_label.config(image=photo)
    image_label.image = photo

def detect_alphabet():
    input_path = input_image_path.get()
    if input_path:
        input_image = cv2.imread(input_path)
        input_image = cv2.resize(input_image, (100, 100))
        preprocessed_templates = preprocess_templates(template_dir)
        recognized_alphabets = recognize_alphabets(input_image, preprocessed_templates)

        result_text = "Recognized Alphabets:\n"
        for i, (alphabet, mse) in enumerate(recognized_alphabets):
            result_text += f"Segment {i+1}: {alphabet} (MSE={mse})\n"
        
        result_label.config(text=result_text)
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

# Add a label for displaying the results
result_label = tk.Label(result_frame, text="Recognized Alphabets:")
result_label.pack()

# Add a button to perform alphabet detection
detect_button = tk.Button(root, text="Detect Alphabet", command=detect_alphabet)
detect_button.pack(pady=10)

# Directory containing template alphabet images
template_dir = 'templates'  # Adjust this directory as needed

# Start the main event loop
root.mainloop()
