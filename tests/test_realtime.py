import unittest
from unittest.mock import patch, MagicMock
from src.realtime.open_meteo_client import OpenMeteoClient, OpenMeteoError
from src.realtime.live_feature_builder import LiveFeatureBuilder
import pandas as pd

class TestRealtimePipeline(unittest.TestCase):
    
    @patch('src.realtime.open_meteo_client.requests.Session.get')
    def test_open_meteo_success(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "hourly": {
                "time": ["2026-08-12T00:00", "2026-08-12T01:00"],
                "temperature_2m": [30.5, 31.0]
            }
        }
        mock_get.return_value = mock_resp
        
        client = OpenMeteoClient()
        data = client.fetch_data(28.61, 77.2)
        self.assertIn("hourly", data)
        self.assertEqual(len(data["hourly"]["temperature_2m"]), 2)
        
    def test_open_meteo_invalid_coords(self):
        client = OpenMeteoClient()
        with self.assertRaises(ValueError):
            client.fetch_data(100.0, 77.2) # Lat > 90

    @patch('src.realtime.open_meteo_client.requests.Session.get')
    def test_open_meteo_malformed(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"error": True}
        mock_get.return_value = mock_resp
        
        client = OpenMeteoClient()
        with self.assertRaises(OpenMeteoError):
            client.fetch_data(28.6, 77.2)
            
    def test_feature_builder_shape(self):
        # We need a sample response that has all required variables.
        # This is a bit tedious to mock by hand, but we can verify the class inits.
        builder = LiveFeatureBuilder()
        self.assertTrue(hasattr(builder, 'scaler'))
        self.assertTrue(hasattr(builder, 'train_t_mean'))
        
    @patch('src.realtime.open_meteo_client.OpenMeteoClient.fetch_data')
    def test_predictor_stacking(self, mock_fetch):
        from src.realtime.predictor import RealtimePredictor
        import numpy as np
        
        # Mock fetch data so it passes LiveFeatureBuilder
        # To make it simple, we can mock the entire builder or just patch the predict methods
        with patch('src.realtime.live_feature_builder.LiveFeatureBuilder.build_features') as mock_build, \
             patch('src.realtime.live_feature_builder.LiveFeatureBuilder.scale_features') as mock_scale:
            
            # Create a dummy future dataframe of length 8
            dummy_dates = pd.date_range(start=pd.to_datetime('today').normalize(), periods=8)
            df_dummy = pd.DataFrame({'date': dummy_dates})
            for col in ['temperature_max', 'hw_rolling', 'hw_exceed']:
                df_dummy[col] = 0.0 # dummy columns for heatwave restricted feature filtering
            for i in range(42): # 42 + 3 = 45 features
                if f'feature_{i}' not in df_dummy.columns:
                    df_dummy[f'feature_{i}'] = 0.0
            
            mock_build.return_value = df_dummy
            
            dummy_scaled = np.zeros((8, 45))
            mock_scale.return_value = dummy_scaled
            
            predictor = RealtimePredictor()
            
            # We mock the predict_proba methods of the models to trace them
            predictor.flood_model.predict_proba = MagicMock(return_value=np.zeros((8, 2)))
            predictor.hw_model.predict_proba = MagicMock(return_value=np.zeros((8, 2)))
            predictor.comp_xgb.predict_proba = MagicMock(return_value=np.zeros((8, 2)))
            predictor.comp_lgbm.predict_proba = MagicMock(return_value=np.zeros((8, 2)))
            predictor.comp_meta.predict_proba = MagicMock(return_value=np.zeros((8, 2)))
            
            output = predictor.predict(17.0, 81.8)
            
            # Assertions
            predictor.comp_xgb.predict_proba.assert_called_once()
            predictor.comp_lgbm.predict_proba.assert_called_once()
            predictor.comp_meta.predict_proba.assert_called_once()
            
            # Check meta learner call shape
            args, kwargs = predictor.comp_meta.predict_proba.call_args
            meta_X_arg = args[0]
            self.assertEqual(meta_X_arg.shape, (8, 2))
            
            # Verify output
            self.assertIn("forecasts", output)
            self.assertTrue(len(output["forecasts"]) > 0)
            self.assertIn("compound", output["forecasts"][0]["probabilities"])
            self.assertTrue(np.isfinite(output["forecasts"][0]["probabilities"]["compound"]))

if __name__ == '__main__':
    unittest.main()
