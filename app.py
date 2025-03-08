from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the model
try:
    model = pickle.load(open('california_housing_model.pkl', 'rb'))
except FileNotFoundError:
    print("Error: Model file 'california_housing_model.pkl' not found. Make sure the model training script has been run.")
    exit()
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

@app.route('/')
def index():
    return render_template('index.html')  # Serve the HTML file

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request (assuming JSON format)
        data = request.get_json()

        # Ensure all required features are present
        required_features = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
        if not all(feature in data for feature in required_features):
            return jsonify({'error': 'Missing required features.  Requires: {}'.format(required_features)}), 400

        # Extract feature values from the JSON data
        features = [data['MedInc'], data['HouseAge'], data['AveRooms'], data['AveBedrms'], data['Population'], data['AveOccup'], data['Latitude'], data['Longitude']]

        # Convert features to a numpy array and reshape
        features = np.array(features).reshape(1, -1) # Reshape is CRUCIAL

        # Make prediction
        prediction = model.predict(features)[0]  # Extract the scalar prediction

        # Return the prediction as JSON
        return jsonify({'prediction': prediction})

    except Exception as e:
        return jsonify({'error': str(e)}), 500  # Handle errors more gracefully


if __name__ == '__main__':
    app.run(debug=True)  # Development server.  Set debug=False for production.