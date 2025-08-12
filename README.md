#  Elastic Net Regression – House Price Prediction

## Overview

This project demonstrates **House Price Prediction** using the **Elastic Net Regression** algorithm. Elastic Net combines **L1 (Lasso)** and **L2 (Ridge)** penalties, making it useful when:

  - You have many correlated features.
  - You want both feature selection and regularization.

The model is trained on a dataset containing **200 rows** of housing data, then deployed using a **Flask web app** with HTML & CSS for the user interface.

-----

## Features

  - **Elastic Net Regression** model for prediction.
  - **Flask backend** for deployment.
  - **HTML/CSS frontend** for user input & output display.
  - Dataset with 200 housing records for training.

-----

## Project Structure

```
elastic_net_house_price/
│
├── model.py             # Trains and saves the model
├── app.py               # Flask app for prediction
├── templates/
│   ├── index.html       # Input form
│   └── result.html      # Prediction result
├── static/
│   └── style.css        # Styles for the frontend
├── dataset.csv          # Dataset for training
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

-----

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file contains:

```
Flask==3.0.0
numpy==1.26.2
pandas==2.1.4
scikit-learn==1.3.2
```

-----

## Dataset

The dataset (`dataset.csv`) contains 200 rows of housing features.

Example:

```
bedrooms,bathrooms,sqft_living,sqft_lot,floors,price
3,2,1800,5000,1,450000
4,3,2200,7500,2,550000
```

Columns:

  - `bedrooms`: Number of bedrooms
  - `bathrooms`: Number of bathrooms
  - `sqft_living`: Living area in square feet
  - `sqft_lot`: Lot area in square feet
  - `floors`: Number of floors
  - `price`: House price (target)

-----

## How It Works

### Model Training (`model.py`)

  - Loads and preprocesses the dataset.
  - Trains an Elastic Net Regression model.
  - Saves the model as `model.pkl`.

### Prediction (`app.py`)

  - Loads `model.pkl`.
  - Accepts input from the HTML form.
  - Predicts house price.
  - Displays the result.

-----

## Running the Project

1.  **Train the model:**
    ```bash
    python model.py
    ```
2.  **Start Flask app:**
    ```bash
    python app.py
    ```
3.  **Open in browser:**
    `http://127.0.0.1:5000/`

-----

## Screenshots
---
Home Page
<img width="528" height="469" alt="Screenshot 2025-08-12 122427" src="https://github.com/user-attachments/assets/5ca41368-00ff-4109-9c2a-ad1cb3612d9f" />

---
Prediction Result

<img width="546" height="522" alt="Screenshot 2025-08-12 122436" src="https://github.com/user-attachments/assets/18d6f5d7-0a50-466b-8e5e-9914bfc8649e" />


