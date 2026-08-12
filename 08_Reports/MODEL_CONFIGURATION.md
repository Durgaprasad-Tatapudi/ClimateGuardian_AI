# Class Imbalance Report
- **flood_target**: 29.56% positive rate in Train.
- **heatwave_target**: 6.93% positive rate in Train.
  *Highly imbalanced.* Applied adaptive class weights safely within training.
- **compound_target**: 3.98% positive rate in Train.
  *Highly imbalanced.* Applied adaptive class weights safely within training.

## Experiment Protocol
- **Train Period**: 2000-2012
- **Validation Period**: 2013-2015
- **Test Period**: 2016-2018
- **Hyperparameter Tuning**: RandomizedSearchCV via TimeSeriesSplit (3 folds, 2 iterations).
- **Deep Learning**: Sequence Length=14, Batch Size=32, Max Epochs=15, Early Stopping Patience=5.
- **Class Imbalance Strategy**: scale_pos_weight/class_weights for ML, pos_weight in BCEWithLogitsLoss for DL.
- **Evaluation Metrics**: F1-Score, ROC-AUC, Brier Score calculated ONLY on chronological Test bounds.
