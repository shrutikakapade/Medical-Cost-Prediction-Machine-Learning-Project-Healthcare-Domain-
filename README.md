<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    
</head>
<body>

<h1>🏥 Medical Cost Prediction – Healthcare Domain</h1>
<h3>A Machine Learning & Streamlit Dashboard Project</h3>

<div class="section">
    <p>
        This project focuses on predicting <strong>individual medical insurance charges</strong> using demographic, lifestyle,
        and health-related variables. The workflow includes <strong>Exploratory Data Analysis (EDA)</strong>,
        <strong>interactive visualization using Streamlit</strong>, and <strong>Machine Learning model development</strong> to estimate healthcare costs accurately.
    </p>
</div>

<div class="section">
    <h2>📌 Project Overview</h2>
    <p>
        Health insurance companies often struggle to determine accurate premiums due to limited individual risk data.
        This project aims to:
    </p>
    <ul>
        <li>Identify key factors influencing medical costs</li>
        <li>Build complete EDA & visualization pipeline</li>
        <li>Develop ML models to predict charges</li>
        <li>Deploy an interactive <strong>Streamlit Dashboard</strong> for data exploration</li>
    </ul>
</div>

<div class="section">
    <h2>📊 Dataset Information</h2>
    <table border="1" cellpadding="10" cellspacing="0">
        <tr><th>Feature</th><th>Description</th></tr>
        <tr><td>age</td><td>Age of primary beneficiary</td></tr>
        <tr><td>sex</td><td>Gender (male/female)</td></tr>
        <tr><td>bmi</td><td>Body Mass Index</td></tr>
        <tr><td>children</td><td>Number of dependents</td></tr>
        <tr><td>smoker</td><td>Smoking status</td></tr>
        <tr><td>region</td><td>US region of beneficiary</td></tr>
        <tr><td>charges</td><td>Medical cost billed (Target)</td></tr>
    </table>
</div>


<div class="section">
    <h2>🔍 Sprint 1 – Exploratory Data Analysis (EDA)</h2>
    <h3>Key tasks completed:</h3>
    <ul>
        <li>Distribution analysis for all features</li>
        <li>Correlation analysis with respect to <strong>charges</strong></li>
        <li>Outlier detection & handling</li>
        <li>Business insights & recommendations</li>
    </ul>
<div
    <h3>Major Insights</h3>
    <ul>
        <li><strong>Smoking</strong> is the strongest predictor of medical cost</li>
        <li><strong>BMI</strong> has a positive correlation—higher BMI leads to higher charges</li>
        <li><strong>Age</strong> contributes significantly after 40+</li>
        <li>Children and region have minimal impact</li>
    </ul>
<div
    <h3>Business Recommendations</h3>
    <ul>
        <li>Higher premiums for smokers due to higher claim risks</li>
        <li>Discounts for individuals maintaining healthy BMI</li>
        <li>Age-based premium bands for better risk evaluation</li>
    </ul>
</div>

<div class="section">
    <h2>📈 Sprint 2 – Interactive Streamlit Dashboard</h2>
    <p>The dashboard includes:</p>
    <ul>
        <li>Feature distribution analysis</li>
        <li>Scatterplots, heatmaps, boxplots</li>
        <li>Interactive filters for dynamic insights</li>
    </ul>
<div
    <h3>Run the Dashboard</h3>
<pre>
streamlit run app.py
</pre>
</div>

<div class="section">
    <h2>🤖 Sprint 3 – Model Building</h2>
<div
    <h3>Algorithms Used</h3>
    <ul>
        <li>K-Nearest Neighbors</li>
        <li>Linear Regression</li>
        <li>Support Vector Regression</li>
        <li>Decision Tree Regressor</li>
        <li>Random Forest Regressor</li>
    </ul>
<div
    <h3>Data Preparation</h3>
    <ul>
        <li>Train-test split: <strong>75% / 25%</strong></li>
        <li>Standardization for numerical features</li>
        <li>One-Hot Encoding for categorical features</li>
        <li>Evaluation Metric: <strong>Mean Absolute Error (MAE)</strong></li>
    </ul>
<div
    <h3>Model Comparison</h3>
    <p><strong>Random Forest</strong> achieved the lowest MAE and performed best overall.</p>
</div>

<div class="section">
    <h2>🏆 Final Outcome</h2>
    <ul>
        <li>Accurate ML model for predicting medical charges</li>
        <li>Interactive Streamlit dashboard for exploration</li>
        <li>Complete ML pipeline from EDA → Modeling → Evaluation</li>
        <li>Useful business insights for insurance premium pricing</li>
    </ul>
</div>

<div class="section">
    <h2>🛠 Tech Stack</h2>
    <ul>
        <li>Python</li>
        <li>Pandas, NumPy</li>
        <li>Matplotlib, Seaborn</li>
        <li>Scikit-Learn</li>
        <li>Streamlit</li>
    </ul>
</div>




<div class="section">
    <h2>👩‍💻 Author</h2>
    <p><strong>Shrutika Kapade</strong><br>
    B.Tech in Data Science | ML & AI Enthusiast | Innomatics Intern</p>
</div>

<div class="footer">
    ⭐ If you like this project, don't forget to star the repository!
</div>

</body>
</html>

