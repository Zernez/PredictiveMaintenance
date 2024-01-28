import pandas as pd
from io import StringIO

# Assuming your data is provided as a string
data = """
C1 (high) Bearing 1 1 S1 50 123 580
C1 (high) Bearing 1 1 S2 50 123 580
C1 (high) Bearing 1 2 S1 120 161 580
C1 (high) Bearing 1 2 S2 110 161 580
C1 (high) Bearing 1 3 S1 100 158 580
C1 (high) Bearing 1 3 S2 100 158 580
C1 (high) Bearing 1 4 S1 50 122 580
C1 (high) Bearing 1 4 S2 50 122 580
C1 (high) Bearing 1 5 S1 30 52 580
C1 (high) Bearing 1 5 S2 30 52 580
C2 (medium) Bearing 2 1 S1 150 491 703
C2 (medium) Bearing 2 1 S2 120 491 703
C2 (medium) Bearing 2 2 S1 60 161 703
C2 (medium) Bearing 2 2 S2 60 161 703
C2 (medium) Bearing 2 3 S1 50 533 703
C2 (medium) Bearing 2 3 S2 130 533 703
C2 (medium) Bearing 2 4 S1 20 42 703
C2 (medium) Bearing 2 4 S2 20 42 703
C2 (medium) Bearing 2 5 S1 110 339 703
C2 (medium) Bearing 2 5 S2 160 339 703
C3 (low) Bearing 3 1 S1 90 2538 878
C3 (low) Bearing 3 1 S2 180 2538 878
C3 (low) Bearing 3 2 S1 150 2496 878
C3 (low) Bearing 3 2 S2 310 2496 878
C3 (low) Bearing 3 3 S1 260 371 878
C3 (low) Bearing 3 3 S2 80 371 878
C3 (low) Bearing 3 4 S1 20 1515 878
C3 (low) Bearing 3 4 S2 290 1515 878
C3 (low) Bearing 3 5 S1 20 114 878
C3 (low) Bearing 3 5 S2 70 114 878
"""

# Create a DataFrame from the provided data
df = pd.read_csv(StringIO(data), delimiter=' ', skipinitialspace=True, header=None)

# Rename columns for clarity
df.columns = ["Condition", "Text", "Dataset", "Bearing1", "Bearing2", "Bearing3", "EventTime", "ActualEventTime", "L10m"]

# Compute the difference between the third and fourth columns
df["Difference"] = df["EventTime"] - df["ActualEventTime"]

# Compute the pct difference between the third and fourth columns
df["DifferencePct"] = ((df["ActualEventTime"] -  df["EventTime"]) /  df["EventTime"]) * 100

# Display the DataFrame with the added "Difference" column
print(df[['Condition', 'Difference', 'DifferencePct']])