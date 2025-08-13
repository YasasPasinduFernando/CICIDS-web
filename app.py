import os
from flask import Flask, request, render_template, send_file
import pandas as pd
import numpy as np
import joblib

UPLOAD_FOLDER = 'uploads'
MODEL_PATH = 'models/cicids_ensemble.pkl'
RESULTS_FOLDER = 'results'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Load model
artifacts = joblib.load(MODEL_PATH)
model = artifacts['ensemble']
scaler = artifacts['scaler']
feature_names = artifacts['feature_names']

def preprocess_csv(file_path):
    df = pd.read_csv(file_path, low_memory=False)
    # Drop non-ML columns
    drop_cols = ['Flow ID','FlowID','Src IP','Source IP','Dst IP','Destination IP','Timestamp','Label']
    for c in drop_cols:
        if c in df.columns:
            if c != 'Label':
                df.drop(columns=c, inplace=True)
    # Numeric conversion
    for col in df.columns:
        if col != 'Label':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    
    Xn = df.select_dtypes(include=['number'])
    for missing in set(feature_names) - set(Xn.columns):
        Xn[missing] = 0
    Xn = Xn[feature_names]
    Xn_s = scaler.transform(Xn)
    proba = model.predict_proba(Xn_s)[:,1]
    pred = (proba >= 0.5).astype(int)
    df['attack_prob'] = proba
    df['prediction(1=ATTACK)'] = pred
    return df

@app.route('/', methods=['GET', 'POST'])
def index():
    download_link = None
    tables = None
    filename = None

    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = file.filename
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(path)
            df_result = preprocess_csv(path)
            
            # Save result CSV
            result_path = os.path.join(app.config['RESULTS_FOLDER'], f'predicted_{filename}')
            df_result.to_csv(result_path, index=False)
            
            tables = [df_result.head().to_html(classes='data')]
            download_link = f'/download/{os.path.basename(result_path)}'

    return render_template('index.html', tables=tables, filename=filename, download_link=download_link)

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join(app.config['RESULTS_FOLDER'], filename),
                     mimetype='text/csv',
                     download_name=filename,
                     as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
