import sys
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os

model = joblib.load("models/model.joblib")

if len(sys.argv) > 1:
    input_file = sys.argv[1]
    df = pd.read_csv(input_file)
else:
    df = pd.read_csv("data/processed/features.csv")

household_ids = df["household"] if "household" in df.columns else df.index

X = df.drop(columns=["label", "household"], errors="ignore")

predictions = model.predict(X)

results = []
for i, pred in enumerate(predictions):
    household_id = household_ids.iloc[i] if hasattr(household_ids, "iloc") else household_ids[i]
    status = "Theft" if pred == 1 else "Normal"
    print(f"Household {household_id}: {status}")
    results.append({"household": household_id, "status": status})

os.makedirs("data/processed", exist_ok=True)
results_df = pd.DataFrame(results)
results_df.to_csv("data/processed/predictions.csv", index=False)
print("\n Predictions saved to data/processed/predictions.csv")

plt.figure(figsize=(10, 5))
colors = ["red" if s["status"] == "Theft" else "green" for s in results]
plt.bar(results_df["household"], [1 if s["status"] == "Theft" else 0 for s in results],
        color=colors)
plt.xlabel("Household ID")
plt.ylabel("Status (0 = Normal, 1 = Theft)")
plt.title("Energy Theft Detection Results")
plt.show()
