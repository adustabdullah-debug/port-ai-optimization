# Create the data generator
cat > data/data_generator.py << 'EOF'
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_synthetic_port_data():
    """Generate synthetic port data that mimics POLA operations"""
    np.random.seed(42)
    
    # Generate date range
    dates = pd.date_range(start='2022-01-01', end='2023-06-30', freq='D')
    
    data = []
    for date in dates:
        # Base patterns
        day_of_week = date.weekday()
        is_weekend = 1 if day_of_week >= 5 else 0
        month = date.month
        
        # Seasonal patterns (higher in summer/fall)
        seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * (month - 7) / 12)
        
        # Generate features
        vessels_at_anchor = max(0, np.random.poisson(8 * seasonal_factor))
        terminal_throughput = max(10000, np.random.normal(20000, 5000) * seasonal_factor)
        
        # Vessel wait time (target variable)
        base_wait = 30 + vessels_at_anchor * 2.5
        weather_impact = np.random.exponential(5)
        wait_time = max(5, np.random.normal(base_wait + weather_impact, 8))
        
        # Truck turn time
        base_turn = 60 + (vessels_at_anchor / 10) * 15
        turn_time = max(15, np.random.normal(base_turn, 12))
        
        data.append({
            'date': date,
            'day_of_week': day_of_week,
            'month': month,
            'is_weekend': is_weekend,
            'is_holiday': 0,
            'vessels_at_anchor': vessels_at_anchor,
            'terminal_throughput': terminal_throughput,
            'vessel_wait_time': wait_time,
            'truck_turn_time': turn_time,
            'wind_speed': np.random.gamma(2, 2),
            'visibility': np.random.uniform(5, 15),
            'precipitation': np.random.exponential(0.5)
        })
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    df = generate_synthetic_port_data()
    df.to_csv('data/synthetic_port_data.csv', index=False)
    print("Synthetic data generated successfully!")
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
EOF