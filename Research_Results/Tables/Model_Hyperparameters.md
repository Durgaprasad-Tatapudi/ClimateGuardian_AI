| Model | Target | Key Hyperparameters |
|-------|--------|---------------------|
| RandomForest | Flood | `n_estimators=100`, `max_depth=5`, `class_weight=balanced` |
| LightGBM | Heatwave | `n_estimators=50`, `max_depth=-1`, `learning_rate=0.1`, restricted features |
| XGBoost | Compound (Base) | `n_estimators=100`, `max_depth=3`, `learning_rate=0.01` |
| LightGBM | Compound (Base) | `n_estimators=50`, `max_depth=-1`, `learning_rate=0.1` |
| LogisticRegression | Compound (Meta) | `C=1.0`, `solver=lbfgs`, `class_weight=balanced` |
| GRU | Compound (Offline) | `hidden_size=64`, `num_layers=2`, `dropout=0.2`, `seq_len=14` |
