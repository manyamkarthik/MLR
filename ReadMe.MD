# California Housing Price Prediction Web App

This project implements a simple web application using Flask and a pre-trained machine learning model to predict California housing prices. The app uses a multi-linear regression model trained on the California Housing dataset. Users can input various features of a house and its location, and the app will predict the median house value for that area.

## Features

*   **Interactive Web Interface:** Uses HTML, CSS, and JavaScript to provide a user-friendly web form for inputting house features.
*   **Detailed Input Descriptions:**  Each input field has a tooltip and associated paragraph that explains the purpose and units of the feature.
*   **Real-Time Prediction:** Predicts the house price based on the provided inputs and displays the result in a formatted currency.
*   **Data Visualization:**  Uses Chart.js to generate a bar chart visualizing the predicted price.
*   **Error Handling:** Provides basic error handling for missing inputs or issues with the prediction.
*   **Clean Codebase:** Well-structured Python code with comments for better readability and maintainability.

## Technologies Used

*   **Flask:** A Python web framework for building the web application.
*   **Scikit-learn:** A Python library for machine learning, used for training and loading the prediction model.
*   **Pandas:** Used for data manipulation during model training.
*   **NumPy:**  Used for numerical operations and array manipulation.
*   **Chart.js:** A JavaScript library for creating interactive charts and visualizations.
*   **HTML/CSS/JavaScript:** For building the user interface.

## Setup and Installation

1.  **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Install the required Python packages:**

    ```bash
    pip install flask scikit-learn pandas numpy
    ```

3.  **Train the Model (if necessary):**

    If you don't have the `california_housing_model.pkl` file, you need to train the model using the provided training script.

    ```bash
    python train_model.py  # Creates california_housing_model.pkl
    ```

    *Make sure that the `train_model.py` is in the same directory as `app.py` or adjust the file path accordingly.*

4.  **Run the Flask Application:**

    ```bash
    python app.py
    ```

    This will start the Flask development server, usually on `http://127.0.0.1:5000/`.

5.  **Access the Web App:**

    Open your web browser and go to `http://127.0.0.1:5000/` to access the application.

## Usage

1.  Enter the values for each feature in the provided form. Tooltips and descriptions are available to help understand the meaning of each input.
2.  Click the "Predict Price" button.
3.  The predicted median house value will be displayed below the form, along with a bar chart visualization.

## File structure
california-housing-prediction/
├── app.py # Flask application file
├── train_model.py # Python script to train the ML model
├── california_housing_model.pkl # Trained ML model (output of train_model.py)
├── templates/ # Directory for HTML templates
│ └── index.html # HTML file for the web interface
└── README.md # This README file


## Feature Descriptions

Here's a brief overview of the features used in the prediction model:

*   **MedInc:** Median income for households within a block of houses (in tens of thousands of US Dollars). Higher values generally indicate more affluent areas.
*   **HouseAge:** Median age of houses within a block. A lower number indicates newer buildings.
*   **AveRooms:** Average number of rooms per household within a block. A higher number often indicates larger homes.
*   **AveBedrms:** Average number of bedrooms per household within a block.
*   **Population:** Total number of people residing within a block. Higher population density can affect house prices.
*   **AveOccup:** Average number of people per household within a block.
*   **Latitude:** A measure of how far north a house is. Higher values represent locations farther north.
*   **Longitude:** A measure of how far west a house is. Higher values represent locations farther west.

## Potential Improvements

*   **Advanced Model:** Implement more sophisticated machine learning models for improved prediction accuracy (e.g., Random Forest, Gradient Boosting).
*   **Ocean Proximity Handling:** Incorporate the `oceanProximity` feature, which is categorical.  This would require encoding the categorical data (e.g., using one-hot encoding).
*   **Data Preprocessing:** Implement more robust data preprocessing techniques, such as scaling or normalization.
*   **Interactive Map:** Integrate a map using Leaflet or Google Maps to visualize the location of the house and surrounding area.
*   **CORS Configuration:** For a production deployment, configure CORS properly to restrict access to your API.
*   **Deployment:** Deploy the app to a cloud platform like AWS, Google Cloud, or Azure.
*   **More Chart Types:**  Allow the user to select different types of charts (e.g., pie chart, line chart).
*   **Responsive Design:** Improve the responsiveness of the web interface for different screen sizes.
*   **Input Validation:**  Add client-side and server-side input validation to prevent invalid data from being submitted.
*   **Model Evaluation:** Incorporate model evaluation metrics (e.g., Mean Squared Error, R-squared) into the web interface.

