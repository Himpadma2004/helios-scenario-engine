"""
Scenario Sensitivity & Stress Response Engine — Prediction API
Project HELIOS | AI Model 1

Flask server that loads trained XGBoost models and exposes a REST API
for real-time what-if scenario predictions.

Run:  python api_server.py
URL:  http://localhost:5000
"""

import os
import json
import numpy as np
import xgboost as xgb
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# ── Paths ────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
FRONTEND   = os.path.join(BASE_DIR, 'frontend')


# ── Portable Scaler (no pickle needed) ───────────────────────────────────
class PortableScaler:
    """StandardScaler loaded from JSON — version independent."""
    def __init__(self, json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        self.mean_ = np.array(data['mean'])
        self.scale_ = np.array(data['scale'])

    def transform(self, X):
        return (np.array(X) - self.mean_) / self.scale_


# ── Load Models & Artifacts ──────────────────────────────────────────────
def load_artifacts():
    """Load all trained models, scaler, and metadata (portable format)."""
    # Metadata
    with open(os.path.join(OUTPUT_DIR, 'model_metadata.json'), 'r') as f:
        metadata = json.load(f)

    # Scaler (from JSON)
    scaler = PortableScaler(os.path.join(OUTPUT_DIR, 'scaler.json'))

    # Models (from native XGBoost format — version independent)
    models = {}
    for target in metadata['target_columns']:
        booster = xgb.Booster()
        booster.load_model(os.path.join(OUTPUT_DIR, f'model_{target}.xgb'))
        models[target] = booster

    # Feature importance
    importance = {}
    imp_path = os.path.join(OUTPUT_DIR, 'feature_importance.csv')
    if os.path.exists(imp_path):
        import csv
        with open(imp_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                target = row['Target']
                if target not in importance:
                    importance[target] = {}
                importance[target][row['Feature']] = float(row['Importance'])

    return metadata, scaler, models, importance


print("Loading model artifacts...")
metadata, scaler, models, importance_data = load_artifacts()
print(f"✅ Loaded {len(models)} models: {list(models.keys())}")


def compute_interaction_features(base_input):
    """Compute interaction features from 6 base inputs. Server-side only."""
    out = dict(base_input)
    out['Fuel_x_Carbon'] = out['Avg_Fuel_Price_Index'] * out['Carbon_Tax_per_Ton']
    out['Renewable_Shortfall'] = max(0.0, 1.0 - out['Avg_Renewable_Generation_Factor'])
    out['Shortfall_x_RPO'] = out['Renewable_Shortfall'] * out['RPO_Target_pct']
    out['VarCost_x_Inversion'] = out['Avg_Variable_Cost_per_MWh'] * out['Merit_Order_Inversion_Rate']
    return out


# ── Flask App ────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=FRONTEND)
CORS(app)


@app.route('/')
def serve_frontend():
    """Serve the frontend HTML."""
    return send_from_directory(FRONTEND, 'index.html')


@app.route('/<path:path>')
def serve_static(path):
    """Serve static frontend files."""
    return send_from_directory(FRONTEND, path)


@app.route('/api/metadata', methods=['GET'])
def get_metadata():
    """Return model metadata including feature ranges and performance."""
    return jsonify(metadata)


@app.route('/api/importance', methods=['GET'])
def get_importance():
    """Return feature importance data for all targets."""
    return jsonify(importance_data)


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Predict system outcomes for a given scenario.

    Expected JSON body (6 base features — interactions computed server-side):
    {
        "Avg_Fuel_Price_Index": 150.0,
        "Carbon_Tax_per_Ton": 500.0,
        "Avg_Renewable_Generation_Factor": 0.85,
        "RPO_Target_pct": 40.0,
        "Avg_Variable_Cost_per_MWh": 120.0,
        "Merit_Order_Inversion_Rate": 0.3
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON body provided'}), 400

        base_features = metadata.get('base_features', metadata['feature_columns'])
        missing = [f for f in base_features if f not in data]
        if missing:
            return jsonify({'error': f'Missing features: {missing}'}), 400

        # Compute interaction features server-side
        full_input = compute_interaction_features(data)
        feature_cols = metadata['feature_columns']
        X_input = np.array([[full_input[f] for f in feature_cols]])
        X_scaled = scaler.transform(X_input)

        # Predict each target
        dmat = xgb.DMatrix(X_scaled)
        predictions = {}
        for target, model in models.items():
            pred = model.predict(dmat)[0]
            predictions[target] = round(float(pred), 4)

        return jsonify({
            'predictions': predictions,
            'input': data,
            'status': 'ok'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/stress_test', methods=['POST'])
def stress_test():
    """
    Run multiple scenarios in batch for stress curve visualization.

    Expected JSON body:
    {
        "base_scenario": { ... 7 features ... },
        "sweep_feature": "Carbon_Tax_per_Ton",
        "sweep_min": 0,
        "sweep_max": 2000,
        "sweep_steps": 20
    }
    """
    try:
        data = request.get_json()
        base = data['base_scenario']
        sweep_feat = data['sweep_feature']
        sweep_min = float(data['sweep_min'])
        sweep_max = float(data['sweep_max'])
        steps = int(data.get('sweep_steps', 20))

        feature_cols = metadata['feature_columns']
        sweep_values = np.linspace(sweep_min, sweep_max, steps).tolist()

        results = []
        for val in sweep_values:
            scenario = dict(base)
            scenario[sweep_feat] = val
            full_input = compute_interaction_features(scenario)
            X_input = np.array([[full_input[f] for f in feature_cols]])
            X_scaled = scaler.transform(X_input)

            dmat = xgb.DMatrix(X_scaled)
            preds = {}
            for target, model in models.items():
                preds[target] = round(float(model.predict(dmat)[0]), 4)

            results.append({
                'sweep_value': round(val, 2),
                'predictions': preds
            })

        return jsonify({
            'sweep_feature': sweep_feat,
            'results': results,
            'status': 'ok'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/compare', methods=['POST'])
def compare_scenarios():
    """
    Compare two scenarios side-by-side.
    
    Expected JSON:
    {
        "scenario_a": { ... 7 features ... },
        "scenario_b": { ... 7 features ... }
    }
    """
    try:
        data = request.get_json()
        feature_cols = metadata['feature_columns']
        
        results = {}
        for label in ['scenario_a', 'scenario_b']:
            scenario = data[label]
            full_input = compute_interaction_features(scenario)
            X_input = np.array([[full_input[f] for f in feature_cols]])
            X_scaled = scaler.transform(X_input)
            dmat = xgb.DMatrix(X_scaled)
            
            preds = {}
            for target, model in models.items():
                preds[target] = round(float(model.predict(dmat)[0]), 4)
            results[label] = preds
        
        # Compute deltas
        deltas = {}
        for target in metadata['target_columns']:
            a_val = results['scenario_a'][target]
            b_val = results['scenario_b'][target]
            deltas[target] = {
                'absolute': round(b_val - a_val, 4),
                'pct_change': round(((b_val - a_val) / abs(a_val)) * 100, 2) if a_val != 0 else None
            }
        
        return jsonify({
            'scenario_a': results['scenario_a'],
            'scenario_b': results['scenario_b'],
            'deltas': deltas,
            'status': 'ok'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok', 'models': len(models), 'version': metadata.get('version', 'unknown')})


# ── Run ──────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print("\n" + "="*60)
    print("  HELIOS Scenario Sensitivity Engine — API Server")
    print("="*60)
    print(f"  Frontend:  http://localhost:{port}")
    print(f"  API Docs:  POST /api/predict, /api/stress_test, /api/compare")
    print(f"  Metadata:  GET  /api/metadata, /api/importance")
    print("="*60 + "\n")
    app.run(host='0.0.0.0', port=port, debug=False)
