import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class OpenMeteoError(Exception):
    pass

class OpenMeteoClient:
    """
    Client for fetching weather data from the official Open-Meteo API.
    Implements retries, timeouts, and response validation.
    """
    
    BASE_URL = "https://api.open-meteo.com/v1/forecast"
    
    def __init__(self, timeout: int = 10, max_retries: int = 3):
        self.timeout = timeout
        
        # Setup session with retry logic
        self.session = requests.Session()
        retries = Retry(
            total=max_retries,
            backoff_factor=1,
            status_forcelist=[408, 429, 500, 502, 503, 504],
            allowed_methods=["GET"]
        )
        adapter = HTTPAdapter(max_retries=retries)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def fetch_data(self, lat: float, lon: float, past_days: int = 14, forecast_days: int = 7) -> Dict[str, Any]:
        """
        Fetches hourly variables for past and future days to construct historical lags and live forecasts.
        """
        if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
            raise ValueError(f"Invalid coordinates: lat={lat}, lon={lon}")

        # The internal model requires the following aggregated features:
        # temperature_avg, temperature_min, temperature_max, dewpoint, dewpoint_min, dewpoint_max,
        # rainfall, runoff, soil_moisture_layer_1, surface_pressure, u_component, v_component, total_evaporation
        
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": [
                "temperature_2m",
                "relative_humidity_2m",
                "dew_point_2m",
                "precipitation",
                "surface_pressure",
                "wind_speed_10m",
                "wind_direction_10m",
                "soil_moisture_0_to_1cm",
                "runoff",
                "evapotranspiration"
            ],
            "past_days": past_days,
            "forecast_days": forecast_days,
            "timezone": "auto"
        }
        
        try:
            response = self.session.get(self.BASE_URL, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            
            # Validation
            if "hourly" not in data or "time" not in data["hourly"]:
                raise OpenMeteoError("Malformed response: missing 'hourly' data.")
                
            return data
            
        except requests.exceptions.Timeout:
            logger.error("Open-Meteo API request timed out.")
            raise OpenMeteoError("Request timed out.")
        except requests.exceptions.RequestException as e:
            logger.error(f"Open-Meteo API request failed: {str(e)}")
            raise OpenMeteoError(f"API request failed: {str(e)}")
