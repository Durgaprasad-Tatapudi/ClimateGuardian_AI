import json
import logging
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import joblib

from .open_meteo_client import OpenMeteoClient
from .live_feature_builder import LiveFeatureBuilder

logger = logging.getLogger(__name__)

MODELS_DIR = Path("05_Models_corrected")

class RealtimePredictor:
    def __init__(self):
        self.client = OpenMeteoClient()
        self.builder = LiveFeatureBuilder()
        
        # Load Frozen Models based on FINAL_MODEL_SELECTION
        # Flood: RandomForest
        self.flood_model = joblib.load(MODELS_DIR / "RF_flood_target.joblib")
        # Heatwave: LightGBM
        self.hw_model = joblib.load(MODELS_DIR / "LGBM_heatwave_target.joblib")
        # Compound: Stacking (XGB, LGBM -> LR)
        self.comp_xgb = joblib.load(MODELS_DIR / "XGB_compound_target.joblib")
        self.comp_lgbm = joblib.load(MODELS_DIR / "LGBM_compound_target.joblib")
        self.comp_meta = joblib.load(MODELS_DIR / "LR_meta_compound_target.joblib")
        
        # Define thresholds derived from 90th percentile / F1-optimization in historical
        # In a full system, these are read from validation metrics
        self.thresholds = {
            "flood": {"Low": 0.3, "Medium": 0.6, "High": 0.85},
            "heatwave": {"Low": 0.2, "Medium": 0.5, "High": 0.8},
            "compound": {"Low": 0.1, "Medium": 0.3, "High": 0.6}
        }
        
    def _get_risk_level(self, prob: float, hazard: str) -> str:
        thresh = self.thresholds[hazard]
        if prob >= thresh["High"]:
            return "High"
        elif prob >= thresh["Medium"]:
            return "Medium"
        elif prob >= thresh["Low"]:
            return "Low"
        return "Minimal"

    def predict(self, lat: float, lon: float) -> dict:
        """
        Executes the live operational pipeline.
        """
        # 1. Fetch
        logger.info(f"Fetching data for {lat}, {lon}")
        raw_data = self.client.fetch_data(lat, lon)
        
        # 2. Build Features
        df_features = self.builder.build_features(raw_data)
        
        # Filter to only Today and Future (horizons: 0, 1, 3, 5, 7)
        today = pd.to_datetime('today').normalize()
        df_future = df_features[df_features['date'] >= today].copy()
        df_future = df_future.reset_index(drop=True)
        
        # 3. Scale Features
        X_scaled = self.builder.scale_features(df_future)
        
        # 4. Predict
        flood_probs = self.flood_model.predict_proba(X_scaled)[:, 1]
        
        # Heatwave uses restricted features
        hw_features = [c for c in df_future.drop(columns=['date']).columns if not ('temperature_max' in c or 'hw_rolling' in c or 'hw_exceed' in c)]
        col_indices = [df_future.drop(columns=['date']).columns.get_loc(c) for c in hw_features]
        hw_probs = self.hw_model.predict_proba(X_scaled[:, col_indices])[:, 1]
        
        # Compound Stacking
        xgb_prob = self.comp_xgb.predict_proba(X_scaled)[:, 1]
        lgbm_prob = self.comp_lgbm.predict_proba(X_scaled)[:, 1]
        
        meta_X = np.column_stack([xgb_prob, lgbm_prob])
        comp_probs = self.comp_meta.predict_proba(meta_X)[:, 1]
        
        # 5. Format Output
        prediction_time = datetime.now(timezone.utc).isoformat()
        results = []
        
        horizons_to_extract = [0, 1, 3, 5, 7]
        for h in horizons_to_extract:
            if h < len(df_future):
                f_date = df_future.loc[h, 'date']
                f_prob = float(flood_probs[h])
                h_prob = float(hw_probs[h])
                c_prob = float(comp_probs[h])
                
                res = {
                    "horizon_days": h,
                    "forecast_date": f_date.isoformat().split("T")[0],
                    "probabilities": {
                        "flood": f_prob,
                        "heatwave": h_prob,
                        "compound": c_prob
                    },
                    "risk_levels": {
                        "flood": self._get_risk_level(f_prob, "flood"),
                        "heatwave": self._get_risk_level(h_prob, "heatwave"),
                        "compound": self._get_risk_level(c_prob, "compound")
                    },
                    "debug": {
                        "xgb_probability": float(xgb_prob[h]),
                        "lgbm_probability": float(lgbm_prob[h]),
                        "meta_input_shape": [1, 2]
                    }
                }
                results.append(res)
                
        output = {
            "location": {"latitude": lat, "longitude": lon},
            "prediction_time": prediction_time,
            "data_source": "Open-Meteo",
            "model_version": "v1.0-frozen-research",
            "forecasts": results
        }
        
        return output
