# Pokémon Battle Predictor

Pick two Pokémon and an XGBoost model predicts the winner. The page shows who won, the model's verdict, a win-probability bar and a stat comparison.

Files:
- `app.py`: the Flask app. It does the same feature engineering as the notebook and loads the model.
- `xgboost_model.json`: the trained model.
- `Data/Pokemon.csv`: the Pokémon dataset (the same file the notebook uses).
- `requirements.txt`, `render.yaml`: tell Render how to install and run it.

Run locally: `pip install -r requirements.txt`, then `python app.py`.
