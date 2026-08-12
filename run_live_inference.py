import json
import argparse
from src.realtime.predictor import RealtimePredictor

def main():
    parser = argparse.ArgumentParser(description="Live Inference Pipeline for ClimateGuardian")
    parser.add_argument("--lat", type=float, required=True, help="Latitude")
    parser.add_argument("--lon", type=float, required=True, help="Longitude")
    args = parser.parse_args()
    
    predictor = RealtimePredictor()
    result = predictor.predict(args.lat, args.lon)
    
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
