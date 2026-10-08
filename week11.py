import pandas as pd
import matplotlib.pylot as plt

df = pd.read_csv("Iris.csv")

print(df.head())
print(df.info())
print(df.describe())

df.hist()
plt.show()

df.plot(kind="box")
plt.show()

pd.plotting.scatter_matrix(df)
plt.show()
