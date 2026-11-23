import pandas as pd
import matplotlib.pyplot as plt

# Show all columns, no scientific notation
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.float_format", "{:.1f}".format)

df = pd.read_csv("gym_members.csv")

print(df.shape)
print(df.describe())

plt.scatter(df["visits_per_month"], df["monthly_extras_spend"], alpha=0.6)
plt.xlabel("Visits per month")
plt.ylabel("Monthly extras spend ($)")
plt.title("Extras spend vs visits")
plt.savefig("scatter.png")
print("Saved scatter.png")