import pandas as pd

data = {
    "A": [],
    "B": []
}

for i in range(3):
    data["A"].append(i)
    data["B"].append(0)

for i in range(2):
    data["A"].append(i)
    data["B"].append(1)

df = pd.DataFrame(data=data)
print(df)

b_labels = df["B"].unique()
print(b_labels)

df_b_0 = df[df["B"] == 0]
print(df_b_0)
