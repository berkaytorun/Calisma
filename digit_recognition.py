import numpy as np
from flask import Flask, request, render_template, jsonify
from PIL import Image, ImageOps, ImageEnhance
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Initialize Flask app
app = Flask(__name__)

# Load the digit dataset and train the model
digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits['data'], digits['target'], test_size=0.3, random_state=42)
model = LogisticRegression(solver='lbfgs', max_iter=1000)
model.fit(X_train, y_train)

# Route for the homepage to upload an image
@app.route('/')
def index():
    return render_template('upload.html')

# Route for checking if the server is running
@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({"message": "Server is running!"})

# Route for uploading an image and getting a prediction
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    
    try:
        # Open the uploaded image and convert to grayscale
        img = Image.open(file).convert('L')
        img.save("static/original_uploaded_image.png")  # Save original image

        # Resize to a larger size first
        img = img.resize((32, 32), Image.LANCZOS)
        img.save("static/resized_large_image.png")

        # Enhance contrast
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(2.0)
        img.save("static/enhanced_image.png")

        # Resize to 8x8 pixels
        img = img.resize((8, 8), Image.LANCZOS)
        img.save("static/resized_image.png")

        # Invert colors
        img = ImageOps.invert(img)
        img.save("static/inverted_image.png")

        # Normalize pixel values
        img_array = np.array(img) / 255.0 * 16.0
        img_array = img_array.flatten().reshape(1, -1)

        # Save processed image
        processed_image = Image.fromarray(np.uint8(img_array.reshape(8, 8) * 255 / 16.0))
        processed_image.save("static/processed_digit.png")
        
        # Predict the digit
        prediction = model.predict(img_array)[0]
        
        # Render the prediction page
        return render_template('result.html', prediction=prediction)

    except Exception as e:
        return jsonify({"error": str(e)})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)