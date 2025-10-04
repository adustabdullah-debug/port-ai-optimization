cat > src/data_processor.py << 'EOF'
import pandas as pd
import numpy as np

class DataProcessor:
    def __init__(self):
        pass
    
    def load_and_clean_data(self, file_path):
        """Load and clean the port operations data"""
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])
        
        # Handle missing values
        df = df.fillna(method='ffill')
        
        return df
    
    def engineer_features(self, df):
        """Create additional features for the model"""
        # Rolling averages
        df['vessel_wait_7d_avg'] = df['vessel_wait_time'].rolling(7).mean()
        df['throughput_7d_avg'] = df['terminal_throughput'].rolling(7).mean()
        
        # Lag features
        df['vessels_anchor_lag1'] = df['vessels_at_anchor'].shift(1)
        df['wait_time_lag1'] = df['vessel_wait_time'].shift(1)
        
        return df
EOF