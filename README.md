# Masters Decision Support System

An interactive, data-driven machine learning application deployed via Streamlit to assist engineering candidates in evaluating whether to pursue a Master's degree. The application builds an end-to-end classification workflow—leveraging historical academic-professional profiles to map non-linear decision boundaries between standardized qualification tiers and continuous salary matrices.

## Dataset Description
The machine learning pipeline operates on a tabular profile dataset (`dataset.csv`), evaluating features across historical student instances:
* **Target Label:** `Should_Do_Masters` (Binary decision outcome encoded as `1` for Yes or `0` for No).
* **Predictor Features:** * `GATE_Score`: Graded engineering entry qualification tiers.
  * `Salary`: Current professional industry salary distribution scaled continuously in INR.

## Methodology
1. **Data Preprocessing & Encoding:** Utilized Scikit-learn's `LabelEncoder` to translate multi-class categorical qualification strings and target variables into standardized numeric tracking matrices.
2. **Data Partitioning:** Split feature matrices into an 80% training slice and a 20% test slice to maintain unbiased evaluation parameters.
3. **Model Selection & Architecture:** Configured and fit an ensemble `RandomForestClassifier` operating with 500 independent decision trees to capture multi-feature interactions smoothly.
4. **Dashboard Engineering:** Built a 5-page fully functional interface utilizing `Streamlit` and `streamlit_option_menu` containing Home, About Model, Dataset, Prediction Engine, and Visual Analytics views.

## Results & Visual Analytics
The application encapsulates predictive capability alongside interactive data discovery surfaces. The web application natively processes inputs and renders real-time Matplotlib plots including:
* Categorical bar distributions for qualification scores.
* Matplotlib histogram charts displaying salary variance scales.
* Distribution pie charts mapping decision outcome ratios inside the UI container.

## How to Run Locally
1. Clone this repository to your computer workspace environment.
2. Install standard package dependencies via pip:
   ```bash
   pip install numpy pandas matplotlib scikit-learn streamlit streamlit-option-menu
