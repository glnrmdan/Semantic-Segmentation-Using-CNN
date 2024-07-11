# from flask import Flask, request, render_template
# from util import NeuralNetwork, binary_cross_entropy
# import numpy as np
# import cv2
# import os
# from PIL import Image

# app = Flask(__name__)

# # Load model weights
# def load_weights(filename):
#     with open(filename, 'rb') as f:
#         data = np.load(f)
#         weights_hidden2_output = data['weights_hidden2_output']
#         bias_hidden2_output = data['bias_hidden2_output']
#         weights_hidden1_hidden2 = data['weights_hidden1_hidden2']
#         bias_hidden1_hidden2 = data['bias_hidden1_hidden2']
#         weights_input_hidden1 = data['weights_input_hidden1']
#         bias_input_hidden1 = data['bias_input_hidden1']
#     return weights_hidden2_output, bias_hidden2_output, weights_hidden1_hidden2, bias_hidden1_hidden2, weights_input_hidden1, bias_input_hidden1

# # Initialize the neural network
# input_size = 3
# hidden_size1 = 3
# hidden_size2 = 2
# output_size = 2

# weights_hidden2_output, bias_hidden2_output, weights_hidden1_hidden2, bias_hidden1_hidden2, weights_input_hidden1, bias_input_hidden1 = load_weights("models/6. Bobot_skenario_6.npz")
# model_loaded = NeuralNetwork(input_size, hidden_size1, hidden_size2, output_size)
# model_loaded.weights_hidden2_output = weights_hidden2_output
# model_loaded.bias_hidden2_output = bias_hidden2_output
# model_loaded.weights_hidden1_hidden2 = weights_hidden1_hidden2
# model_loaded.bias_hidden1_hidden2 = bias_hidden1_hidden2
# model_loaded.weights_input_hidden1 = weights_input_hidden1
# model_loaded.bias_input_hidden1 = bias_input_hidden1

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/upload', methods=['POST'])
# def upload_file():
#     if 'file' not in request.files:
#         return 'No file part'
#     file = request.files['file']
#     if file.filename == '':
#         return 'No selected file'
#     if file:
#         # Save uploaded image
#         uploaded_image_path = os.path.join('static', 'uploaded_image.png')
#         file.save(uploaded_image_path)

#         # Process image
#         img = Image.open(uploaded_image_path)
#         img = np.array(img)
#         input_image_reshaped = cv2.resize(img, (128, 128)) / 255.0
#         features = input_image_reshaped.reshape(-1, 3)

#         # Predict segmentation mask
#         predicted_mask = model_loaded.forward(features)
#         predicted_mask = predicted_mask.argmax(axis=1).reshape(128, 128)
#         segmented_image = np.zeros((128, 128, 3), dtype=np.uint8)

#         vegetation_color = [0, 255, 0]  # Green
#         non_vegetation_color = [0, 0, 255]  # Blue
#         segmented_image[predicted_mask == 0] = non_vegetation_color
#         segmented_image[predicted_mask == 1] = vegetation_color

#         # Save segmented image
#         output_image = Image.fromarray(segmented_image)
#         segmented_image_path = os.path.join('static', 'segmented_image.png')
#         output_image.save(segmented_image_path)

#         return render_template('results.html', uploaded_image_path=uploaded_image_path, segmented_image_path=segmented_image_path)

# if __name__ == '__main__':
#     app.run(debug=True)

# COBA
from flask import Flask, request, render_template, url_for
from util import NeuralNetwork, binary_cross_entropy
import numpy as np
import cv2
import os
from PIL import Image
import shutil

app = Flask(__name__)

# Load model weights
def load_weights(filename):
    with open(filename, 'rb') as f:
        data = np.load(f)
        weights_hidden2_output = data['weights_hidden2_output']
        bias_hidden2_output = data['bias_hidden2_output']
        weights_hidden1_hidden2 = data['weights_hidden1_hidden2']
        bias_hidden1_hidden2 = data['bias_hidden1_hidden2']
        weights_input_hidden1 = data['weights_input_hidden1']
        bias_input_hidden1 = data['bias_input_hidden1']
    return weights_hidden2_output, bias_hidden2_output, weights_hidden1_hidden2, bias_hidden1_hidden2, weights_input_hidden1, bias_input_hidden1

# Initialize the neural network
input_size = 3
hidden_size1 = 3
hidden_size2 = 2
output_size = 2

weights_hidden2_output, bias_hidden2_output, weights_hidden1_hidden2, bias_hidden1_hidden2, weights_input_hidden1, bias_input_hidden1 = load_weights("models/6. Bobot_skenario_6.npz")
model_loaded = NeuralNetwork(input_size, hidden_size1, hidden_size2, output_size)
model_loaded.weights_hidden2_output = weights_hidden2_output
model_loaded.bias_hidden2_output = bias_hidden2_output
model_loaded.weights_hidden1_hidden2 = weights_hidden1_hidden2
model_loaded.bias_hidden1_hidden2 = bias_hidden1_hidden2
model_loaded.weights_input_hidden1 = weights_input_hidden1
model_loaded.bias_input_hidden1 = bias_input_hidden1

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        # Save uploaded image
        filename = file.filename
        uploaded_image_path = os.path.join('static', 'uploads', filename)

        # Ensure the upload directory exists
        if not os.path.exists(os.path.dirname(uploaded_image_path)):
            os.makedirs(os.path.dirname(uploaded_image_path))

        file.save(uploaded_image_path)

        # Process image
        img = Image.open(uploaded_image_path)
        img = np.array(img)
        input_image_reshaped = cv2.resize(img, (128, 128)) / 255.0
        features = input_image_reshaped.reshape(-1, 3)

        # Predict segmentation mask
        predicted_mask = model_loaded.forward(features)
        predicted_mask = predicted_mask.argmax(axis=1).reshape(128, 128)
        segmented_image = np.zeros((128, 128, 3), dtype=np.uint8)

        vegetation_color = [0, 255, 0]  # Green
        non_vegetation_color = [0, 0, 255]  # Blue
        segmented_image[predicted_mask == 0] = non_vegetation_color
        segmented_image[predicted_mask == 1] = vegetation_color

        # Save segmented image
        segmented_image_path = os.path.join('static', 'segmented', filename)

        # Ensure the segmented directory exists
        if not os.path.exists(os.path.dirname(segmented_image_path)):
            os.makedirs(os.path.dirname(segmented_image_path))

        output_image = Image.fromarray(segmented_image)
        output_image.save(segmented_image_path)

        # Find and copy ground truth image
        ground_truth_dir = os.path.join('patching_fix', 'patch_ground')
        ground_truth_image_path = os.path.join(ground_truth_dir, filename)

        if os.path.exists(ground_truth_image_path):
            ground_truth_dest_path = os.path.join('static', 'ground_truth', filename)

            # Ensure the ground truth directory exists
            if not os.path.exists(os.path.dirname(ground_truth_dest_path)):
                os.makedirs(os.path.dirname(ground_truth_dest_path))

            shutil.copy(ground_truth_image_path, ground_truth_dest_path)
        else:
            return "Ground truth image not found", 404

        return render_template('results.html', 
                               uploaded_image_path=url_for('static', filename=f'uploads/{filename}'), 
                               segmented_image_path=url_for('static', filename=f'segmented/{filename}'),
                               ground_truth_image_path=url_for('static', filename=f'ground_truth/{filename}'))

if __name__ == '__main__':
    app.run(debug=True)