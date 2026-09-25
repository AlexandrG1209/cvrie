import csv
import pandas as pd


def load_dataset(filename): # Load the patient dataset and return it as a pandas DataFrame.
    
    rows = []
    with open(filename, "r", encoding="utf-8-sig") as file:
        reader = csv.reader(file)

        for row in reader:

            if len(row) == 2:
                rows.append({
                    "id": int(row[0]),
                    "color": None,
                    "text": row[1]
                })

            elif len(row) == 3:
                rows.append({
                    "id": int(row[0]),
                    "color": row[1],
                    "text": row[2]
                })

    return pd.DataFrame(rows)