cat > src/model_trainer.py << 'EOF'
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import joblib

class ModelTrainer:
    def __init__(self):
        self.model = None
        self.feature_names = []
    
    def prepare_features(self, df):
        """Prepare features for model training"""
        features = ['day_of_week', 'month', 'is_weekend', 'is_holiday',
                   'vessels_at_anchor', 'terminal_throughput', 
                   'wind_speed', 'visibility', 'precipitation',
                   'vessel_wait_7d_avg', 'throughput_7d_avg',
                   'vessels_anchor_lag1', 'wait_time_lag1']
        
        # Use only features that exist in dataframe
        available_features = [f for f in features if f in df.columns]
        self.feature_names = available_features
        
        X = df[available_features].fillna(0)
        y = df['vessel_wait_time']
        
        return X, y
    
    def train_model(self, X, y):
        """Train XGBoost model"""
        # Split data chronologically
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Train model
        self.model = XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        
        print(f"Train R²: {train_r2:.3f}")
        print(f"Test R²: {test_r2:.3f}")
        print(f"Test MAE: {test_mae:.2f} hours")
        
        return test_r2, test_mae
    
    def plot_feature_importance(self):
        """Plot feature importance"""
        if self.model is None:
            print("Model not trained yet!")
            return
        
        importance = self.model.feature_importances_
        feature_names = self.feature_names
        
        # Create plot
        plt.figure(figsize=(10, 6))
        indices = np.argsort(importance)[::-1]
        plt.barh(range(len(indices)), importance[indices])
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.title('Feature Importance - Vessel Wait Time Prediction')
        plt.tight_layout()
        plt.savefig('../results/feature_importance.png')
        plt.show()
    
    def save_model(self, file_path):
        """Save trained model"""
        if self.model is None:
            print("No model to save!")
            return
        
        joblib.dump(self.model, file_path)
        print(f"Model saved to {file_path}")
EOF