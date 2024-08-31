import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk
import os

class AlphabetDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Alphabet Detection")
        
        # Created a frame for the title Alphabet Detection
        self.title_frame = tk.Frame(root)
        self.title_frame.pack(pady=10)

        # Added a label for the title
        self.title_label = tk.Label(self.title_frame, text="Alphabet Detection", font=("Arial", 18))
        self.title_label.pack()

        # Created a frame for the input image selection
        self.input_frame = tk.Frame(root)
        self.input_frame.pack(pady=10)

        # Added a label and button for selecting the input image
        self.input_label = tk.Label(self.input_frame, text="Select Input Image:")
        self.input_label.grid(row=0, column=0)

        self.input_image_path = tk.StringVar()
        self.input_entry = tk.Entry(self.input_frame, textvariable=self.input_image_path, width=40)
        self.input_entry.grid(row=0, column=1)

        self.browse_button = tk.Button(self.input_frame, text="Browse", command=self.select_image)
        self.browse_button.grid(row=0, column=2)

        # Created a frame for displaying the selected image
        self.image_frame = tk.Frame(root)
        self.image_frame.pack(pady=10)

        # Added a label for displaying the selected image
        self.image_label = tk.Label(self.image_frame)
        self.image_label.pack()

        # Created a frame for displaying the results
        self.result_frame = tk.Frame(root)
        self.result_frame.pack(pady=10)

        # Added labels for displaying the results
        self.result_label = tk.Label(self.result_frame, text="Recognized Alphabet:")
        self.result_label.pack()

        self.mse_label = tk.Label(self.result_frame, text="Minimum MSE:")
        self.mse_label.pack()

        self.similarity_label = tk.Label(self.result_frame, text="Max Similarity Percentage:")
        self.similarity_label.pack()

        # Added a button to perform alphabet detection
        self.detect_button = tk.Button(root, text="Detect Alphabet", command=self.detect_alphabet)
        self.detect_button.pack(pady=10)

        # Directory containing template alphabet images
        self.template_dir = 'templates'  # Adjust this directory as needed

    def show(self,title,image):
        cv2.imshow(title,image)
        cv2.waitKey(0)

    def select_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
        if file_path:
            self.input_image_path.set(file_path)
            self.load_image(file_path)

    def load_image(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (300, 300))
        photo = ImageTk.PhotoImage(image=Image.fromarray(image))
        self.image_label.config(image=photo)
        self.image_label.image = photo

    def detect_alphabet(self):
        input_path = self.input_image_path.get()
        if input_path:
            preprocessed_templates = self.preprocess_templates(self.template_dir)
            input_image=cv2.imread(input_path)
            recognized_alphabet, min_mse, max_similarity_percentage = self.recognize_alphabet(input_image, preprocessed_templates)
            self.result_label.config(text=f"Recognized Alphabet: {recognized_alphabet}")
            self.mse_label.config(text=f"Minimum MSE: {min_mse}")
            self.similarity_label.config(text=f"Max Similarity Percentage: {max_similarity_percentage}")
        else:
            print("Please select an input image.")

    def preprocess_templates(self, template_dir):
        preprocessed_templates = {}
        for alphabet_file in os.listdir(template_dir):
            template_path = os.path.join(template_dir, alphabet_file)
            template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
            _, thresh = cv2.threshold(template, 1, 255, cv2.THRESH_BINARY)
            preprocessed_templates[alphabet_file] = thresh
        return preprocessed_templates
    
    def compare_images(self, input_processed, template, template_thresh):
        mse = np.mean((input_processed - template_thresh) ** 2)
        non_zero_pixels = np.sum(template_thresh == 255)
        similarity_percentage = (np.sum(template_thresh == input_processed) / non_zero_pixels) * 100
        return mse, similarity_percentage

    def recognize_alphabet(self, input_image, preprocessed_templates):
        input_processed = input_image
        self.show("input",input_processed)
        matched_templates=input_processed
        min_mse = float('inf')
        max_similarity_percentage = 0
        recognized_alphabet = None

        for alphabet_file, template_thresh in preprocessed_templates.items():
            template_mse, similarity_percentage = self.compare_images(input_processed, preprocessed_templates[alphabet_file], template_thresh)
            print(f"Comparing with alphabet '{os.path.splitext(alphabet_file)[0]}': MSE={template_mse}")

            if template_mse < min_mse:
                min_mse = template_mse
                recognized_alphabet = os.path.splitext(alphabet_file)[0]
                matched_templates=template_thresh

            if similarity_percentage > max_similarity_percentage:
                max_similarity_percentage = similarity_percentage

            if min_mse == 0 and max_similarity_percentage == 100:
                break
        
        self.show("matched_templates",matched_templates)
        print("Recognized Alphabet:", recognized_alphabet)
        print("Minimum MSE:", min_mse)
        print("Max Similarity Percentage:", max_similarity_percentage)

        return recognized_alphabet, min_mse, max_similarity_percentage

if __name__ == "__main__":
    root = tk.Tk()
    app = AlphabetDetectionApp(root)
    root.mainloop()
