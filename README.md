# NEER — AI-Powered Water-Quality Monitoring

Neer is an educational software prototype exploring how multiple water-quality measurements can be analysed together using a Random Forest classifier.

## Included
- Streamlit dashboard with four virtual measurement controls.
- Random Forest model trained on a public water-potability dataset.
- Held-out test metrics: accuracy, precision, recall and F1.
- Feature-importance chart and simulated-input analysis.

## Run locally on Windows
1. Open this repository folder in VS Code.
2. Install packages: `py -m pip install -r requirements.txt`
3. Train the model: `py train_model.py`
4. Launch the dashboard: `py -m streamlit run app.py`

The first run downloads the CSV into data/water_quality.csv. The trained model and metrics are saved under model/.

## Deploy
1. Sign in at https://share.streamlit.io/ with GitHub.
2. Choose Create app.
3. Select this repository, branch main, and main file app.py.
4. Deploy. Streamlit installs dependencies from requirements.txt and the app downloads the dataset and trains the model on first launch.

## Dataset
The app uses these four columns: ph, Solids, Conductivity and Turbidity.
Dataset mirror: https://github.com/sudesokin/Water-Quality-Analysis-and-Forecasting/blob/main/dataset/water_potability.csv

The dataset calls its dissolved material feature Solids. Neer preserves that original name and does not claim it is a direct TDS sensor reading.

## Scientific limitations
- This is a software-only demonstration. The sliders provide simulated readings; no physical sensors are connected.
- The model learns labels in the historical dataset. Its output is not a laboratory test or water-safety certification.
- The model score is not the probability that a real sample is safe to drink.
- These four measurements cannot identify specific heavy metals, pesticides or microorganisms.
- Real-world use would require calibrated sensors, representative local data, laboratory reference testing and independent validation.
