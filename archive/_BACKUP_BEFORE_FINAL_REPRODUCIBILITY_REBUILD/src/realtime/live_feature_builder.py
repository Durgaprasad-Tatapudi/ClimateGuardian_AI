import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Paths
FEATURES_DIR = Path("03_Features")
MODELS_DIR = Path("05_Models_corrected")

class LiveFeatureBuilder:
    def __init__(self):
        # We need the training data means for anomalies and the scaler for features.
        self._load_or_create_scaler_and_means()
        
    def _load_or_create_scaler_and_means(self):
        scaler_path = MODELS_DIR / "scaler.joblib"
        means_path = MODELS_DIR / "train_means.joblib"
        
        # In a real environment, these would be exported during the training pipeline.
        # Here we recreate them deterministically from the training set if missing.
        if not scaler_path.exists() or not means_path.exists():
            print("Scaler or means not found, generating from master_features (train period)...")
            df = pd.read_csv(FEATURES_DIR / "master_features.csv")
            df['date'] = pd.to_datetime(df['date'])
            train_mask = (df['date'].dt.year >= 2000) & (df['date'].dt.year <= 2012)
            train_df = df[train_mask]
            
            # Anomaly base variables (as per build_features.py)
            self.train_sm_mean = train_df['soil_moisture_layer_1'].mean()
            self.train_sm_std = train_df['soil_moisture_layer_1'].std()
            self.train_t_mean = train_df['temperature_avg'].mean()
            self.train_p_mean = train_df['surface_pressure'].mean()
            
            joblib.dump({
                "sm_mean": self.train_sm_mean,
                "sm_std": self.train_sm_std,
                "t_mean": self.train_t_mean,
                "p_mean": self.train_p_mean
            }, means_path)
            
            # Prepare X_train for scaler
            feature_cols = [c for c in train_df.columns if c not in ['date']]
            X_train = train_df[feature_cols].copy()
            
            # Impute if needed (training script used SimpleImputer or XGB handled it, wait!
            # train_models.py uses StandardScaler which errors on NaN. Let's check what train_models.py did.
            # train_models.py filled NaN with mean or median). Let's use simple fillna(0) or mean for scaler.
            # Actually train_models.py did: X.fillna(X.mean(), inplace=True)
            self.train_means_for_imputation = X_train.mean()
            X_train.fillna(self.train_means_for_imputation, inplace=True)
            
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
            self.scaler.fit(X_train)
            joblib.dump(self.scaler, scaler_path)
            joblib.dump(self.train_means_for_imputation, MODELS_DIR / "imputation_means.joblib")
            self.feature_cols = feature_cols
            joblib.dump(self.feature_cols, MODELS_DIR / "feature_cols.joblib")
        else:
            self.scaler = joblib.load(scaler_path)
            means = joblib.load(means_path)
            self.train_sm_mean = means["sm_mean"]
            self.train_sm_std = means["sm_std"]
            self.train_t_mean = means["t_mean"]
            self.train_p_mean = means["p_mean"]
            self.train_means_for_imputation = joblib.load(MODELS_DIR / "imputation_means.joblib")
            self.feature_cols = joblib.load(MODELS_DIR / "feature_cols.joblib")

    def build_features(self, open_meteo_data: dict) -> pd.DataFrame:
        """
        Takes raw hourly JSON from Open-Meteo, aggregates to daily,
        builds lags, constructs anomalies, and scales.
        """
        hourly = open_meteo_data["hourly"]
        df_hourly = pd.DataFrame(hourly)
        df_hourly['time'] = pd.to_datetime(df_hourly['time'])
        df_hourly['date'] = df_hourly['time'].dt.date
        
        # Calculate u and v components from wind speed and direction
        # wind_direction is meteorological (0=North, 90=East)
        # u = -ws * sin(dir * pi/180), v = -ws * cos(dir * pi/180)
        ws = df_hourly['wind_speed_10m']
        wd_rad = np.radians(df_hourly['wind_direction_10m'])
        df_hourly['u_component'] = -ws * np.sin(wd_rad)
        df_hourly['v_component'] = -ws * np.cos(wd_rad)
        
        # Aggregate to daily
        daily_aggs = {
            'precipitation': 'sum',
            'runoff': 'sum',
            'evapotranspiration': 'sum',
            'temperature_2m': ['mean', 'min', 'max'],
            'dew_point_2m': ['mean', 'min', 'max'],
            'soil_moisture_0_to_1cm': 'mean',
            'surface_pressure': 'mean',
            'wind_speed_10m': 'mean',
            'u_component': 'mean',
            'v_component': 'mean'
        }
        
        df_daily = df_hourly.groupby('date').agg(daily_aggs)
        df_daily.columns = ['_'.join(col).strip() for col in df_daily.columns.values]
        df_daily = df_daily.reset_index()
        df_daily['date'] = pd.to_datetime(df_daily['date'])
        
        # Variable Mapping
        df_features = pd.DataFrame()
        df_features['date'] = df_daily['date']
        df_features['rainfall'] = df_daily['precipitation_sum']
        df_features['runoff'] = df_daily['runoff_sum']
        df_features['surface_runoff'] = df_daily['runoff_sum'] # Proxy
        df_features['soil_moisture_layer_1'] = df_daily['soil_moisture_0_to_1cm_mean']
        df_features['temperature_avg'] = df_daily['temperature_2m_mean']
        df_features['temperature_min'] = df_daily['temperature_2m_min']
        df_features['temperature_max'] = df_daily['temperature_2m_max']
        df_features['dewpoint'] = df_daily['dew_point_2m_mean']
        df_features['dewpoint_min'] = df_daily['dew_point_2m_min']
        df_features['dewpoint_max'] = df_daily['dew_point_2m_max']
        df_features['surface_pressure'] = df_daily['surface_pressure_mean']
        df_features['u_component'] = df_daily['u_component_mean']
        df_features['v_component'] = df_daily['v_component_mean']
        df_features['wind_speed'] = df_daily['wind_speed_10m_mean']
        df_features['total_evaporation'] = df_daily['evapotranspiration_sum']
        
        # Multi-day and rolling features
        for window in [3, 5, 7, 14]:
            df_features[f'rainfall_{window}d'] = df_features['rainfall'].rolling(window, min_periods=1).sum()
            
        for window in [3, 7]:
            df_features[f'runoff_{window}d'] = df_features['runoff'].rolling(window, min_periods=1).sum()
            
        df_features['soil_moisture_rolling_7d'] = df_features['soil_moisture_layer_1'].rolling(7, min_periods=1).mean()
        df_features['rolling_temperature_3d'] = df_features['temperature_avg'].rolling(3, min_periods=1).mean()
        
        # Anomalies
        df_features['soil_moisture_anomaly'] = (df_features['soil_moisture_layer_1'] - self.train_sm_mean) / self.train_sm_std
        df_features['temperature_anomaly'] = df_features['temperature_avg'] - self.train_t_mean
        df_features['pressure_anomaly'] = df_features['surface_pressure'] - self.train_p_mean
        
        # Temporal
        df_features['month'] = df_features['date'].dt.month
        df_features['day_of_year'] = df_features['date'].dt.dayofyear
        df_features['season'] = df_features['month'].apply(lambda x: 1 if x in [12,1,2] else (2 if x in [3,4,5] else (3 if x in [6,7,8,9] else 4)))
        df_features['monsoon_indicator'] = df_features['month'].isin([6, 7, 8, 9]).astype(int)
        
        # Lags
        lag_cols = ['rainfall', 'runoff', 'temperature_max', 'surface_pressure', 'soil_moisture_layer_1']
        for col in lag_cols:
            for lag in [1, 2, 3]:
                df_features[f'{col}_lag{lag}'] = df_features[col].shift(lag)
                
        # Drop historical rows, keep only today and future
        today = pd.to_datetime('today').normalize()
        # For testing purposes if time is mocked, let's keep all and the predictor will filter by 'today'
        # Or better yet, we just return the full DF and let predictor pick the horizon index.
        
        # Ensure column order matches training exactly
        missing_cols = [c for c in self.feature_cols if c not in df_features.columns]
        for mc in missing_cols:
            df_features[mc] = 0.0 # fallback for missing
            
        return df_features

    def scale_features(self, df_features: pd.DataFrame) -> np.ndarray:
        # Align columns
        X = df_features[self.feature_cols].copy()
        
        # Impute
        X.fillna(self.train_means_for_imputation, inplace=True)
        
        # Scale
        X_scaled = self.scaler.transform(X)
        return X_scaled
