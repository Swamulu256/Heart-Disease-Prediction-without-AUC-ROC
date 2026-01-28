#  Heart Disease Prediction without AUC & ROC – Machine Learning 

This project demonstrates a **Heart Disease Prediction system**  with a simple and clean HTML-based web interface.  
It is designed for learning, demonstration, without advanced evaluation metrics like AUC or ROC curves.

##  Project Overview

Heart disease is one of the leading causes of death worldwide. This project aims to show how basic health parameters can be used to **predict the risk of heart disease** using a simplified prediction logic and a user-friendly web interface.

The focus of this project is:
- Understanding the end-to-end flow of a prediction system
- Connecting health inputs → prediction logic → result display
- Creating a beginner-friendly ML project suitable for interviews and portfolios

##  Prediction Logic

The prediction is based on commonly used health indicators such as:
- Age
- Resting Blood Pressure
- Cholesterol Level
- Maximum Heart Rate

A simple risk score is calculated on the client side to classify the result as:
- Higher Risk of Heart Disease
- Lower Risk of Heart Disease

##  Technologies Used

- **HTML** – Structure of the web page
- **CSS**– Styling and layout
-**python** – Machine Learning model development and training
-   
### Python Libraries

- **NumPy** – Numerical computations and array operations  
- **Pandas** – Data manipulation, cleaning, and analysis  
- **Scikit-learn**– Machine learning algorithms and model training
-   
##  Features
- Simple and clean user interface
- No backend required
- Instant prediction result
- Easy to understand logic
- No AUC, ROC curve, or complex ML metrics

##  How to Run the Project

1. Download or clone this repository
2. Open the `index.html` file
3. Enter the required health details
4. Click on Predict
5. View the prediction result instantly

No installation or setup is required.


##  HTML Code (index.html)

```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Heart Disease Prediction</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body {
      font-family: Arial, Helvetica, sans-serif;
      background: #f4f6f8;
      padding: 40px;
    }
    .container {
      max-width: 600px;
      margin: auto;
      background: #ffffff;
      padding: 25px;
      border-radius: 10px;
      box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    h2 {
      text-align: center;
      margin-bottom: 20px;
    }
    label {
      font-weight: bold;
      margin-top: 10px;
      display: block;
    }
    input, select, button {
      width: 100%;
      padding: 8px;
      margin-top: 5px;
    }
    button {
      margin-top: 20px;
      background: #2563eb;
      color: white;
      border: none;
      border-radius: 6px;
      cursor: pointer;
    }
    button:hover {
      background: #1e40af;
    }
    .result {
      margin-top: 20px;
      font-size: 18px;
      font-weight: bold;
      text-align: center;
    }
  </style>
</head>
<body>

<div class="container">
  <h2>Heart Disease Prediction</h2>

  <label>Age</label>
  <input type="number" id="age" value="50">

  <label>Sex</label>
  <select id="sex">
    <option value="1">Male</option>
    <option value="0">Female</option>
  </select>

  <label>Resting Blood Pressure</label>
  <input type="number" id="bp" value="130">

  <label>Cholesterol</label>
  <input type="number" id="chol" value="240">

  <label>Maximum Heart Rate</label>
  <input type="number" id="hr" value="150">

  <button onclick="predict()">Predict</button>

  <div class="result" id="result"></div>
</div>

<script>
  function predict() {
    const age = Number(document.getElementById('age').value);
    const bp = Number(document.getElementById('bp').value);
    const chol = Number(document.getElementById('chol').value);
    const hr = Number(document.getElementById('hr').value);

    // Simple demo logic (no AUC, no ROC)
    let riskScore = age + bp + chol - hr;

    const result = document.getElementById('result');
    if (riskScore > 350) {
      result.style.color = 'red';
      result.textContent = 'Result: Higher risk of Heart Disease';
    } else {
      result.style.color = 'green';
      result.textContent = 'Result: Lower risk of Heart Disease';
    }
  }
</script>

</body>
</html>


## 📄 License

This project is open-source and available for learning and educational purposes.


