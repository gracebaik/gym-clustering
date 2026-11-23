import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

df = pd.read_csv("gym_members.csv")

# The features k-means will use (member_id is left out on purpose)
features = [
    "age",
    "visits_per_month",
    "avg_visit_minutes",
    "classes_per_month",
    "months_member",
    "monthly_extras_spend",
]

# Step 1: randomly set aside 30% of members as a test set
train, test = train_test_split(df, test_size=0.3, random_state=42)
print("Training members:", train.shape[0])
print("Test members:", test.shape[0])

# Step 2: scale, fitting the scaler on TRAINING members only
scaler = StandardScaler()
X_train = scaler.fit_transform(train[features])
# Test members are scaled with the TRAINING averages/spreads (no refitting)
X_test = scaler.transform(test[features])

# Steps 2 + 3: for each k, fit on training, then score the test members
ks = range(1, 9)
train_spread = []
test_spread = []
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    km.fit(X_train)
    train_spread.append(km.inertia_)
    # Assign each test member to its nearest training center, sum squared distances
    test_spread.append(-km.score(X_test))
    print("k =", k,
          " training spread =", round(train_spread[-1], 1),
          " test spread =", round(test_spread[-1], 1))

# Elbow plot with both curves
plt.figure()
plt.plot(ks, train_spread, marker="o", label="Training members (231)")
plt.plot(ks, test_spread, marker="o", label="Test members (99)")
plt.xlabel("Number of groups (k)")
plt.ylabel("Within-group spread")
plt.title("Elbow plot: training vs test")
plt.legend()
plt.savefig("elbow.png")
print("Saved elbow.png")

# Step 4: fit the chosen model (k = 4) on the training members
k_chosen = 4
model = KMeans(n_clusters=k_chosen, n_init=10, random_state=42)
model.fit(X_train)

# Which group each training member landed in
train = train.copy()
train["cluster"] = model.labels_
print()
print("Members per group (training):")
print(train["cluster"].value_counts().sort_index())

# Group centers converted back to real units (the "average member" of each group)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
centers = pd.DataFrame(scaler.inverse_transform(model.cluster_centers_), columns=features)
print()
print("Group centers (average member of each group):")
print(centers.round(1))

import numpy as np

# Step 5a: names and credit decision for each group
# (Check these match YOUR centers table: group 0 should be the young, $9-spend group)
segment_names = {
    0: "Newcomers",
    1: "Loyal veterans",
    2: "Gym regulars",
    3: "Class lovers",
}
gets_credit = {
    0: True,
    1: False,
    2: False,
    3: False,
}

def segment_action(age, visits_per_month, avg_visit_minutes,
                   classes_per_month, months_member, monthly_extras_spend):
    """Takes one member's features and returns their segment and whether they get the $10 credit."""
    member = pd.DataFrame([[age, visits_per_month, avg_visit_minutes,
                            classes_per_month, months_member, monthly_extras_spend]],
                          columns=features)
    member_scaled = scaler.transform(member)          # scale with the TRAINING scaler
    cluster = model.predict(member_scaled)[0]          # nearest group center
    distances = np.linalg.norm(model.cluster_centers_ - member_scaled, axis=1)
    return cluster, segment_names[cluster], gets_credit[cluster], distances

# The new member from the assignment
cluster, name, credit, distances = segment_action(30, 4, 42, 1, 5, 10)
print()
print("New member - distance to each group center:", distances.round(2))
print("New member - group:", cluster, "(" + name + ")")
print("New member - gets the $10 credit:", credit)

# Step 5b: assign EVERY member (training + test) to a group
everyone = df.copy()
everyone["cluster"] = model.predict(scaler.transform(df[features]))
everyone["segment"] = everyone["cluster"].map(segment_names)

# Finance's rule, member by member: +$5 if they spend under $20, -$10 if $20 or more
everyone["net_if_credit"] = np.where(everyone["monthly_extras_spend"] < 20, 5, -10)

baseline_net = everyone["net_if_credit"].sum()     # credit to everyone
v1_net = everyone.loc[everyone["cluster"].map(gets_credit), "net_if_credit"].sum()  # credit to chosen groups only

print()
print("Per segment (all 330 members):")
summary = everyone.groupby("segment").agg(
    members=("member_id", "count"),
    avg_extras_spend=("monthly_extras_spend", "mean"),
    net_if_given_credit=("net_if_credit", "sum"),
)
print(summary.round(1))
print()
print("Monthly net from the credit - baseline (everyone):", baseline_net)
print("Monthly net from the credit - v1 (Newcomers only):", v1_net)
