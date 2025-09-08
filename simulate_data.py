# simulate_data.py
import numpy as np
import pandas as pd
import os

os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)


def generate_smartgrid_data(n_households=1000, n_days=30, theft_ratio=0.2, random_state=42):
    """
    Generate synthetic smart meter data for energy theft detection.
    
    Args:
        n_households (int): Number of households (meters).
        n_days (int): Number of days for simulation.
        theft_ratio (float): Fraction of households doing theft.
        random_state (int): Random seed for reproducibility.
    
    Returns:
        pd.DataFrame: Simulated dataset with consumption and labels.
    """
    np.random.seed(random_state)
    hours = n_days * 24
    
    data = []
    theft_households = np.random.choice(range(n_households),
                                        size=int(n_households * theft_ratio),
                                        replace=False)
    
    for h in range(n_households):
        base_load = np.random.uniform(0.3, 1.0) 
        daily_pattern = np.sin(np.linspace(0, 24*np.pi, hours)) * np.random.uniform(0.5, 1.5)
        noise = np.random.normal(0, 0.2, hours)
        consumption = base_load + daily_pattern + noise
        consumption = np.clip(consumption, 0, None)
    
        if h in theft_households:

            for _ in range(np.random.randint(2, 5)):  
                start = np.random.randint(0, hours - 12)
                end = start + np.random.randint(6, 12)
                consumption[start:end] *= np.random.uniform(0.1, 0.5)
            
            label = 1
        else:
            label = 0
        
        for t in range(hours):
            data.append([h, t, consumption[t], label])
    
    df = pd.DataFrame(data, columns=["household", "hour", "consumption", "label"])
    return df


if __name__ == "__main__":
    df = generate_smartgrid_data()
    df.to_csv("data/raw/simulated_smartgrid.csv", index=False)
    print("Synthetic smart grid data generated and saved to data/raw/simulated_smartgrid.csv")
    print(df.head())