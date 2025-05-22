from flask import Flask, render_template, request, jsonify, send_file
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from werkzeug.utils import secure_filename
import joblib
import io

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'csv'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('ml_models', exist_ok=True)

global_data = {'X': None, 'y': None}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_data(data):
    try:
        # Check if data is empty
        if data.empty:
            raise ValueError("The uploaded file is empty")
            
        # Check if we have enough columns (at least 2 - features and target)
        if data.shape[1] < 2:
            raise ValueError(f"Expected at least 2 columns (features and target), but got {data.shape[1]} columns")
            
        # Separate features and target
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        
        # Check for missing values
        if X.isna().any().any() or y.isna().any():
            raise ValueError("Dataset contains missing values. Please ensure all values are present.")
            
        # Check if all features are numeric
        if not all(X.dtypes.apply(lambda x: np.issubdtype(x, np.number))):
            raise ValueError("All feature columns must be numeric")
            
        # Check if target is binary (0/1)
        unique_values = y.unique()
        if not all(val in [0, 1] for val in unique_values):
            raise ValueError("Target column must contain only 0 and 1 values")
            
        # Scale the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        return X_scaled, y, scaler
    except Exception as e:
        raise ValueError(f"Error preprocessing data: {str(e)}")

def get_model(algo):
    if algo == 'naive_bayes':
        return GaussianNB()
    elif algo == 'logistic_regression':
        return LogisticRegression(max_iter=1000)
    elif algo == 'knn':
        return KNeighborsClassifier()
    elif algo == 'decision_tree':
        return DecisionTreeClassifier()
    elif algo == 'random_forest':
        return RandomForestClassifier(n_estimators=100)
    elif algo == 'gradient_boosting':
        return GradientBoostingClassifier()
    elif algo == 'mlp':
        return MLPClassifier(max_iter=1000)
    elif algo == 'bagging_rf':
        return BaggingClassifier(RandomForestClassifier(), n_estimators=10)
    elif algo == 'bagging_et':
        from sklearn.ensemble import ExtraTreesClassifier
        return BaggingClassifier(ExtraTreesClassifier(), n_estimators=10)
    elif algo == 'bagging_knn':
        return BaggingClassifier(KNeighborsClassifier(), n_estimators=10)
    elif algo == 'bagging_svc':
        return BaggingClassifier(SVC(probability=True), n_estimators=10)
    elif algo == 'bagging_ridge':
        return BaggingClassifier(RidgeClassifier(), n_estimators=10)
    else:
        return None

def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    accuracy = accuracy_score(y_test, y_pred)
    return precision, recall, f1, accuracy

def load_all_models():
    model_files = {
        'naive_bayes': 'naive_bayes_model.pkl',
        'logistic_regression': 'logistic_regression_model.pkl',
        'knn': 'knn_model.pkl',
        'decision_tree': 'decision_tree_model.pkl',
        'random_forest': 'random_forest_model.pkl',
        'gradient_boosting': 'gradient_boosting_model.pkl',
        'mlp': 'mlp_model.pkl',
        'bagging_rf': 'bagging_rf_model.pkl',
        'bagging_et': 'bagging_et_model.pkl',
        'bagging_knn': 'bagging_knn_model.pkl',
        'bagging_svc': 'bagging_svc_model.pkl',
        'bagging_ridge': 'bagging_ridge_model.pkl',
    }
    models = {}
    for key, fname in model_files.items():
        path = os.path.join('ml_models', fname)
        if os.path.exists(path):
            models[key] = joblib.load(path)
    scaler_path = os.path.join('ml_models', 'scaler.pkl')
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    return models, scaler

@app.route('/')
def home():
    return render_template('system.html')

@app.route('/upload', methods=['POST'])
def upload():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Only CSV files are allowed'}), 400
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        try:
            # Read the CSV file
            data = pd.read_csv(filepath)
            
            # Preprocess the data
            X, y, scaler = preprocess_data(data)
            
            # Store the data and save the scaler
            global_data['X'] = X
            global_data['y'] = y
            joblib.dump(scaler, 'ml_models/scaler.pkl')
            
            # Return success message with data shape information
            return jsonify({
                'message': f'File uploaded and data preprocessed successfully. Dataset contains {X.shape[1]} features and {len(y)} samples.',
                'features': X.shape[1],
                'samples': len(y)
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/run_algorithm', methods=['POST'])
def run_algorithm():
    try:
        algo = request.json.get('algorithm')
        if global_data['X'] is None or global_data['y'] is None:
            return jsonify({'error': 'No data uploaded. Please upload a dataset first.'}), 400
        model = get_model(algo)
        if model is None:
            return jsonify({'error': 'Invalid algorithm selected.'}), 400
        X_train, X_test, y_train, y_test = train_test_split(global_data['X'], global_data['y'], test_size=0.2, random_state=42)
        precision, recall, f1, accuracy = evaluate_model(model, X_train, X_test, y_train, y_test)
        # Save the model
        joblib.dump(model, f'ml_models/{algo}_model.pkl')
        return jsonify({
            'algorithm': algo,
            'precision': f'{precision*100:.2f}',
            'recall': f'{recall*100:.2f}',
            'f1_score': f'{f1*100:.2f}',
            'accuracy': f'{accuracy*100:.2f}'
        })
    except Exception as e:
        return jsonify({'error': f'Error running algorithm: {str(e)}'}), 500

@app.route('/predict_vitals', methods=['POST'])
def predict_vitals():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'Please select a file to upload.'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Please select a file to upload.'}), 400
        # Accept CSV or TXT
        if not (file.filename.endswith('.csv') or file.filename.endswith('.txt')):
            return jsonify({'error': 'Invalid file type. Only CSV or TXT files are allowed.'}), 400
        # Read file into DataFrame, auto-detect delimiter
        try:
            content = file.read().decode('utf-8')
            # Clean up content - remove empty lines and trailing whitespace
            content = '\n'.join(line.strip() for line in content.splitlines() if line.strip())
            # Detect delimiter
            delimiter = ','
            if '\t' in content:
                delimiter = '\t'
            df = pd.read_csv(io.StringIO(content), delimiter=delimiter, header=None)
            # If first row is not numeric, treat as header and skip
            try:
                float(df.iloc[0,0])
            except Exception:
                df = df.iloc[1:]
                df = df.reset_index(drop=True)
            # Ensure all values are numeric
            df = df.apply(pd.to_numeric, errors='coerce')
            if df.isna().any().any():
                return jsonify({'error': 'File contains non-numeric values. Please ensure all values are numeric.'}), 400
        except Exception as e:
            return jsonify({'error': f'Could not read the file: {str(e)}. Please ensure it is a valid CSV or TXT with numeric values.'}), 400
        # Load models and scaler
        models, scaler = load_all_models()
        if not models or scaler is None:
            return jsonify({'error': 'Models or scaler not found. Please train models first with a matching dataset.'}), 400
        # Check feature count
        try:
            X = df.values.astype(float)
            expected_features = scaler.mean_.shape[0]
            if X.shape[1] != expected_features:
                return jsonify({'error': f'Feature mismatch: Your file has {X.shape[1]} columns, but the model expects {expected_features}. Please upload a file with the correct number of columns or retrain your model.'}), 400
            X_scaled = scaler.transform(X)
        except Exception:
            return jsonify({'error': 'Could not process the file. Please ensure all values are numeric and the file matches the model features.'}), 400
        results = []
        for idx, row in enumerate(X_scaled):
            patient_result = {'patient_index': idx+1, 'vitals': df.iloc[idx].tolist(), 'predictions': {}}
            for algo, model in models.items():
                try:
                    pred = model.predict([row])[0]
                    status = 'Patient condition is abnormal detected' if pred == 1 else 'Patient condition is stable detected'
                except Exception:
                    status = 'Prediction error.'
                patient_result['predictions'][algo] = status
            results.append(patient_result)
        return jsonify({'results': results})
    except Exception:
        return jsonify({'error': 'An unexpected server error occurred. Please try again or contact support.'}), 500

# Future: Add endpoints for graph data

if __name__ == '__main__':
    app.run(debug=True)
