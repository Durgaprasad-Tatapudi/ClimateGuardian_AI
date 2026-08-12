# Label Definition
## Heatwave Definition
- **Threshold**: 25.22 °C (90th percentile of daily max temperature from train period 2000-2014).
- **Duration**: >= 3 consecutive days.
- **Baseline**: Training period only, preventing leakage.
- **Variables Used**: `temperature_max_C`.

## Flood Definition
- **Event Definition**: Any day falling within the `start_date` and `end_date` of a verified India flood event in GFD.
- **Spatial Matching**: National level (binary). The available datasets are aggregated to a national daily level, so spatial precision is not supported for historical modeling without distinct sub-regional grids.
- **Positive Class**: 1 if a flood is ongoing anywhere in India on that day.
- **Negative Class**: 0 otherwise.

## Compound Event Definition
- **Temporal Relationship**: Flood occurrence within 7 days of a Heatwave.
- **Justification**: Sequential compound events often manifest as extreme heat causing soil desiccation, followed closely by intense precipitation leading to rapid surface runoff (flooding). A 7-day window captures this synoptic-scale transition.
