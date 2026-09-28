# Week Project – SaaS Churn & Revenue Analytics Platform

# ============================================================
# MODULE 7 — Visualisation
# ============================================================

# Imports Libraries and Load Data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned datasets
customers = pd.read_csv("cleaned_customers.csv")
subscriptions = pd.read_csv("cleaned_subscriptions.csv")
usage = pd.read_csv("cleaned_usage.csv")
tickets = pd.read_csv("cleaned_tickets.csv")

print("Customers shape:", customers.shape)
print("Subscriptions shape:", subscriptions.shape)
print("Usage shape:", usage.shape)
print("Tickets shape:", tickets.shape)


# Check the columns

print("\n--- Customers Columns ---")
print(customers.columns.tolist())

print("\n--- Subscriptions Columns ---")
print(subscriptions.columns.tolist())

print("\n--- Usage Columns ---")
print(usage.columns.tolist())

print("\n--- Tickets Columns ---")
print(tickets.columns.tolist())

# Prepare date columns

customers["SignupDate"] = pd.to_datetime(
    customers["SignupDate"],
    errors="coerce"
)

subscriptions["StartDate"] = pd.to_datetime(
    subscriptions["StartDate"],
    errors="coerce"
)

subscriptions["EndDate"] = pd.to_datetime(
    subscriptions["EndDate"],
    errors="coerce"
)

usage["Month"] = pd.to_datetime(
    usage["Month"],
    errors="coerce"
)

tickets["OpenedDate"] = pd.to_datetime(
    tickets["OpenedDate"],
    errors="coerce"
)

print("\n--- Date Conversion Check ---")

print(
    "Customer SignupDate type:",
    customers["SignupDate"].dtype
)

print(
    "Subscription StartDate type:",
    subscriptions["StartDate"].dtype
)

print(
    "Usage Month type:",
    usage["Month"].dtype
)

print(
    "Ticket OpenedDate type:",
    tickets["OpenedDate"].dtype
)

# Create Customer Subscription View

customer_subscription_view = customers[
    [
        "CustomerID",
        "Industry",
        "Country",
        "AcquisitionChannel"
    ]
].merge(
    subscriptions[
        [
            "CustomerID",
            "SubscriptionID",
            "PlanName",
            "MRR",
            "Status"
        ]
    ],
    on="CustomerID",
    how="inner"
)

print("\n--- Customer Subscription View ---")
print(customer_subscription_view.head())

print(
    "\nShape:",
    customer_subscription_view.shape
)

# Prepare churn flag

customer_subscription_view["ChurnFlag"] = np.where(
    customer_subscription_view["Status"] == "Churned",
    1,
    0
)

print("\n--- Churn Flag Check ---")

print(
    customer_subscription_view[
        ["Status", "ChurnFlag"]
    ].drop_duplicates()
)

# Chart 1: Churn Trend Over Time

# Month

churn_data = subscriptions[
    subscriptions["Status"] == "Churned"
].copy()

churn_data["ChurnMonth"] = (
    churn_data["EndDate"]
    .dt.to_period("M")
    .dt.to_timestamp()
)

monthly_churn = (
    churn_data
    .groupby("ChurnMonth")["CustomerID"]
    .nunique()
    .reset_index(name="ChurnedCustomers")
)

print("\n--- Monthly Churn ---")
print(monthly_churn)

# Chart

plt.figure(figsize=(10, 6))

plt.plot(
    monthly_churn["ChurnMonth"],
    monthly_churn["ChurnedCustomers"],
    marker="o"
)

plt.title("Churn Trend Over Time")
plt.xlabel("Month")
plt.ylabel("Number of Churned Customers")

plt.xticks(rotation=45)
plt.grid(True)

plt.tight_layout()
plt.show()

# Chart 2: Retention Curve

# Create customer cohort month

customers["CohortMonth"] = (
    customers["SignupDate"]
    .dt.to_period("M")
)

print("\n--- Customer Cohort Month ---")

print(
    customers[
        ["CustomerID", "SignupDate", "CohortMonth"]
    ].head(10)
)

# Create usage activity month

usage["ActivityMonth"] = (
    usage["Month"]
    .dt.to_period("M")
)

print("\n--- Usage Activity Month ---")

print(
    usage[
        [
            "CustomerID",
            "Month",
            "ActivityMonth"
        ]
    ].head(10)
)

# Merge Cohort with Usage Activity

customer_cohort = customers[
    ["CustomerID", "CohortMonth"]
].copy()

usage_activity = usage[
    ["CustomerID", "ActivityMonth"]
].copy()

# One customer should count only once per month
usage_activity = usage_activity.drop_duplicates(
    subset=["CustomerID", "ActivityMonth"]
)

cohort_activity = customer_cohort.merge(
    usage_activity,
    on="CustomerID",
    how="inner"
)

print("\n--- Cohort Activity ---")
print(cohort_activity.head(10))

print(
    "\nShape:",
    cohort_activity.shape
)

# Calculate Cohort Index

cohort_activity["CohortIndex"] = (
    (cohort_activity["ActivityMonth"].dt.year
     - cohort_activity["CohortMonth"].dt.year) * 12
    +
    (cohort_activity["ActivityMonth"].dt.month
     - cohort_activity["CohortMonth"].dt.month)
)

# Remove activity before signup
cohort_activity = cohort_activity[
    cohort_activity["CohortIndex"] >= 0
].copy()

print("\n--- Cohort Index ---")

print(
    cohort_activity[
        [
            "CustomerID",
            "CohortMonth",
            "ActivityMonth",
            "CohortIndex"
        ]
    ].head(15)
)

# Calculate retention

cohort_counts = (
    cohort_activity
    .groupby(
        ["CohortMonth", "CohortIndex"]
    )["CustomerID"]
    .nunique()
    .reset_index(name="ActiveCustomers")
)

cohort_sizes = (
    customers
    .groupby("CohortMonth")["CustomerID"]
    .nunique()
)

cohort_counts["CohortSize"] = (
    cohort_counts["CohortMonth"]
    .map(cohort_sizes)
)

cohort_counts["RetentionRate"] = (
    cohort_counts["ActiveCustomers"]
    / cohort_counts["CohortSize"]
)

print("\n--- Cohort Retention ---")

print(cohort_counts.head(20))

# Create retention curve data

retention_table = cohort_counts.pivot_table(
    index="CohortMonth",
    columns="CohortIndex",
    values="RetentionRate"
)

retention_curve = retention_table.mean(
    axis=0,
    skipna=True
)

print("\n--- Retention Curve Data ---")

retention_curve_data = pd.DataFrame({
    "MonthSinceSignup": retention_curve.index,
    "AverageRetentionPercent": (
        retention_curve.values * 100
    ).round(2)
})

print(retention_curve_data)

# Create the chart

plt.figure(figsize=(10, 6))

plt.plot(
    retention_curve_data["MonthSinceSignup"],
    retention_curve_data["AverageRetentionPercent"],
    marker="o"
)

plt.title("Customer Retention Curve")
plt.xlabel("Months Since Signup")
plt.ylabel("Average Retention (%)")

plt.grid(True)

plt.tight_layout()
plt.show()

# Chart 3: Churn by Segment

# Churn Rate by Plan Segment

# Total unique customers in each plan
segment_total = (
    customer_subscription_view
    .groupby("PlanName")["CustomerID"]
    .nunique()
)

# Unique churned customers in each plan
segment_churned = (
    customer_subscription_view[
        customer_subscription_view["ChurnFlag"] == 1
    ]
    .groupby("PlanName")["CustomerID"]
    .nunique()
)

# Combine both results
segment_churn = pd.DataFrame({
    "TotalCustomers": segment_total,
    "ChurnedCustomers": segment_churned
}).fillna(0)

# Calculate churn rate
segment_churn["ChurnRate"] = (
    segment_churn["ChurnedCustomers"]
    / segment_churn["TotalCustomers"]
    * 100
)

# Convert PlanName from index back to a column
segment_churn = segment_churn.reset_index()

print("\n--- Churn Rate by Plan Segment ---")
print(segment_churn)

# Create the chart

plt.figure(figsize=(10, 6))

plt.bar(
    segment_churn["PlanName"],
    segment_churn["ChurnRate"]
)

plt.title("Churn Rate by Plan Segment")
plt.xlabel("Plan")
plt.ylabel("Churn Rate (%)")

plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Find the highest churn segment

highest_churn_plan = segment_churn.loc[
    segment_churn["ChurnRate"].idxmax()
]

print("\n--- Highest Churn Plan ---")
print("Plan:", highest_churn_plan["PlanName"])
print("Churn Rate:", round(highest_churn_plan["ChurnRate"], 2), "%")

# Chart 4: Usage Distribution

# Usage Distribution

# Remove missing values from SessionMinutes
session_data = usage["SessionMinutes"].dropna()

print("\n--- Session Minutes Summary ---")
print(session_data.describe())

# Create the usage distribution chart

plt.figure(figsize=(10, 6))

plt.hist(
    session_data,
    bins=20
)

plt.title("Distribution of Customer Session Minutes")
plt.xlabel("Session Minutes")
plt.ylabel("Number of Records")

plt.tight_layout()
plt.show()

# Usage Distribution Range

print("\n--- Usage Distribution Range ---")
print("Minimum Session Minutes:", session_data.min())
print("Median Session Minutes:", session_data.median())
print("Maximum Session Minutes:", session_data.max())

# Chart 5: Usage vs Churn Relationship

# Prepare usage + churn data

# Select required columns from subscriptions
subscription_status = subscriptions[
    ["CustomerID", "Status"]
].copy()

# Merge customer status with usage data
usage_churn = usage.merge(
    subscription_status,
    on="CustomerID",
    how="inner"
)

# Remove missing SessionMinutes values
usage_churn = usage_churn.dropna(
    subset=["SessionMinutes", "Status"]
)

print("\n--- Usage vs Churn Data ---")
print(
    usage_churn[
        ["CustomerID", "SessionMinutes", "Status"]
    ].head()
)

print("\nShape:", usage_churn.shape)

# Create the comparison chart

plt.figure(figsize=(10, 6))

sns.boxplot(
    data=usage_churn,
    x="Status",
    y="SessionMinutes"
)

plt.title("Session Usage vs Customer Churn Status")
plt.xlabel("Customer Status")
plt.ylabel("Session Minutes")

plt.tight_layout()
plt.show()

# Calculate the actual median usage

usage_by_status = (
    usage_churn
    .groupby("Status")["SessionMinutes"]
    .median()
    .sort_values()
)

print("\n--- Median Session Minutes by Status ---")
print(usage_by_status)

# Chart 6: Ticket Satisfaction Impact

# Ticket Satisfaction Impact

# Select customer status from subscriptions
subscription_status = subscriptions[
    ["CustomerID", "Status"]
].copy()

# Merge ticket data with customer status
ticket_churn = tickets.merge(
    subscription_status,
    on="CustomerID",
    how="inner"
)

# Remove missing satisfaction scores
ticket_churn = ticket_churn.dropna(
    subset=["SatisfactionScore", "Status"]
)

print("\n--- Ticket Satisfaction vs Churn Data ---")
print(
    ticket_churn[
        ["CustomerID", "SatisfactionScore", "Status"]
    ].head()
)

print("\nShape:", ticket_churn.shape)

# Calculate average satisfaction

satisfaction_by_status = (
    ticket_churn
    .groupby("Status")["SatisfactionScore"]
    .mean()
    .sort_values()
)

print("\n--- Average Satisfaction Score by Status ---")
print(satisfaction_by_status.round(2))

# Create the chart

plt.figure(figsize=(10, 6))

sns.boxplot(
    data=ticket_churn,
    x="Status",
    y="SatisfactionScore"
)

plt.title("Ticket Satisfaction Score by Customer Status")
plt.xlabel("Customer Status")
plt.ylabel("Satisfaction Score")

plt.tight_layout()
plt.show()

# Chart 7: Correlation Heatmap

# Correlation Heatmap

# Select numeric variables for correlation analysis
correlation_data = usage[
    [
        "Logins",
        "ActiveUsers",
        "APICalls",
        "SessionMinutes"
    ]
].copy()

print("\n--- Correlation Data ---")
print(correlation_data.head())

# Correlation Matrix

correlation_matrix = correlation_data.corr()

print("\n--- Correlation Matrix ---")
print(correlation_matrix.round(2))

# Create the heatmap

plt.figure(figsize=(10, 7))

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    linewidths=0.5
)

plt.title("Correlation Heatmap of Customer Usage Metrics")
plt.xlabel("Usage Metrics")
plt.ylabel("Usage Metrics")

plt.tight_layout()
plt.show()

# strongest relationship

# Remove the diagonal values because each variable
# has a correlation of 1 with itself

correlation_pairs = (
    correlation_matrix
    .where(
        np.triu(
            np.ones(correlation_matrix.shape),
            k=1
        ).astype(bool)
    )
    .stack()
    .sort_values(
        key=lambda x: x.abs(),
        ascending=False
    )
)

print("\n--- Strongest Correlation Pair ---")
print(correlation_pairs.head(1))


# ============================================================
# MODULE 8 - CUSTOMER SEGMENTATION
# ============================================================

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Customer Subscription Features

customer_subscription = (
    subscriptions
    .groupby("CustomerID")
    .agg(
        TotalMRR=("MRR", "sum"),
        TotalSeats=("Seats", "sum"),
        SubscriptionCount=("SubscriptionID", "nunique")
    )
    .reset_index()
)

print("\n--- Customer Subscription Features ---")
print(customer_subscription.head())

print("\nShape:", customer_subscription.shape)

# duplicate customers

print(
    "\nDuplicate CustomerIDs:",
    customer_subscription["CustomerID"].duplicated().sum()
)

# Subscription Feature Statistics

print("\n--- Subscription Feature Statistics ---")
print(
    customer_subscription[
        ["TotalMRR", "TotalSeats", "SubscriptionCount"]
    ].describe()
)

# Customer Usage Features

customer_usage = (
    usage
    .groupby("CustomerID")
    .agg(
        TotalLogins=("Logins", "sum"),
        AvgActiveUsers=("ActiveUsers", "mean"),
        TotalAPICalls=("APICalls", "sum"),
        AvgSessionMinutes=("SessionMinutes", "mean")
    )
    .reset_index()
)

print("\n--- Customer Usage Features ---")
print(customer_usage.head())

print("\nShape:", customer_usage.shape)

# duplicate CustomerIDs

print(
    "\nDuplicate CustomerIDs:",
    customer_usage["CustomerID"].duplicated().sum()
)

# missing values

print("\n--- Missing Values in Usage Features ---")

print(
    customer_usage[
        [
            "TotalLogins",
            "AvgActiveUsers",
            "TotalAPICalls",
            "AvgSessionMinutes"
        ]
    ].isnull().sum()
)

# usage statistics

print("\n--- Usage Feature Statistics ---")

print(
    customer_usage[
        [
            "TotalLogins",
            "AvgActiveUsers",
            "TotalAPICalls",
            "AvgSessionMinutes"
        ]
    ].describe()
)

# Build Customer-Level Ticket Features

customer_tickets = (
    tickets
    .groupby("CustomerID")
    .agg(
        TicketCount=("TicketID", "nunique"),
        AvgResolutionHours=("ResolutionHours", "mean"),
        AvgSatisfactionScore=("SatisfactionScore", "mean")
    )
    .reset_index()
)

print("\n--- Customer Ticket Features ---")
print(customer_tickets.head())

print("\nShape:", customer_tickets.shape)

# duplicate CustomerIDs

print(
    "\nDuplicate CustomerIDs:",
    customer_tickets["CustomerID"].duplicated().sum()
)

# missing values

print("\n--- Missing Values in Ticket Features ---")

print(
    customer_tickets[
        [
            "TicketCount",
            "AvgResolutionHours",
            "AvgSatisfactionScore"
        ]
    ].isnull().sum()
)

# statistics

print("\n--- Ticket Feature Statistics ---")

print(
    customer_tickets[
        [
            "TicketCount",
            "AvgResolutionHours",
            "AvgSatisfactionScore"
        ]
    ].describe()
)

# Combine All Customer-Level Features

# Start with subscription features
customer_features = customer_subscription.merge(
    customer_usage,
    on="CustomerID",
    how="outer"
)

# Add ticket features
customer_features = customer_features.merge(
    customer_tickets,
    on="CustomerID",
    how="outer"
)

print("\n--- Combined Customer Features ---")
print(customer_features.head())

print("\nShape:", customer_features.shape)

# one row per customer

print(
    "\nDuplicate CustomerIDs:",
    customer_features["CustomerID"].duplicated().sum()
)

# columns

print("\n--- Customer Feature Columns ---")
print(customer_features.columns.tolist())

# missing values

print("\n--- Missing Values ---")
print(customer_features.isnull().sum())

# final customer count

print("\nUnique Customers:", customer_features["CustomerID"].nunique())
print("Rows:", len(customer_features))

# Handle Missing Values

print("\n--- Missing Values Before Cleaning ---")
print(
    customer_features.isnull().sum()
)

# Handle ticket-related missing values

# Customers with no tickets
customer_features["TicketCount"] = (
    customer_features["TicketCount"].fillna(0)
)

customer_features["AvgResolutionHours"] = (
    customer_features["AvgResolutionHours"].fillna(0)
)

# Use median for missing satisfaction scores
satisfaction_median = (
    customer_features["AvgSatisfactionScore"].median()
)

customer_features["AvgSatisfactionScore"] = (
    customer_features["AvgSatisfactionScore"]
    .fillna(satisfaction_median)
)

print("\nMedian Satisfaction Used:",
      round(satisfaction_median, 2))

# Handle missing usage features

usage_features = [
    "TotalLogins",
    "AvgActiveUsers",
    "TotalAPICalls",
    "AvgSessionMinutes"
]

for column in usage_features:
    customer_features[column] = (
        customer_features[column]
        .fillna(customer_features[column].median())
    )
    
# Missing Values After Cleaning

print("\n--- Missing Values After Cleaning ---")
print(
    customer_features.isnull().sum()
)

# Verify the dataset

print("\n--- Final Customer Feature Dataset ---")
print(customer_features.head())

print("\nShape:", customer_features.shape)

print(
    "\nDuplicate CustomerIDs:",
    customer_features["CustomerID"].duplicated().sum()
)

# Select Features and Scale Them

segmentation_features = [
    "TotalMRR",
    "TotalSeats",
    "SubscriptionCount",
    "TotalLogins",
    "AvgActiveUsers",
    "TotalAPICalls",
    "AvgSessionMinutes",
    "TicketCount",
    "AvgResolutionHours",
    "AvgSatisfactionScore"
]

X = customer_features[segmentation_features].copy()

print("\n--- Features Used for Clustering ---")
print(X.columns.tolist())

print("\nShape of X:", X.shape)

# feature values

print("\n--- Feature Sample Before Scaling ---")
print(X.head())

# Scale the features

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\n--- Scaled Feature Sample ---")
print(X_scaled[:5])

# Verify the scaling

print("\n--- Scaled Data Mean ---")
print(X_scaled.mean(axis=0).round(2))

print("\n--- Scaled Data Standard Deviation ---")
print(X_scaled.std(axis=0).round(2))

# Elbow Method

inertia_values = []

k_values = range(2, 11)

for k in k_values:

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    kmeans.fit(X_scaled)

    inertia_values.append(kmeans.inertia_)


print("\n--- Inertia Values ---")

for k, inertia in zip(k_values, inertia_values):
    print(
        "K =", k,
        "| Inertia =", round(inertia, 2)
    )
    
# Plot the Elbow Curve

plt.figure(figsize=(10, 6))

plt.plot(
    list(k_values),
    inertia_values,
    marker="o"
)

plt.title("Elbow Method for Optimal Number of Clusters")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")

plt.xticks(list(k_values))
plt.grid(True)

plt.tight_layout()
plt.show()

# Inertia Reduction

print("\n--- Inertia Reduction ---")

for i in range(1, len(inertia_values)):
    reduction = (
        inertia_values[i - 1]
        - inertia_values[i]
    )

    print(
        f"K {list(k_values)[i-1]} -> "
        f"K {list(k_values)[i]}: "
        f"Reduction = {reduction:.2f}"
    )
    
# Run K-Means Clustering

# Use the K selected from the Elbow Method
selected_k = 4

print("\nSelected K:", selected_k)

# Run K-Means

kmeans_final = KMeans(
    n_clusters=selected_k,
    random_state=42,
    n_init=10
)

cluster_labels = kmeans_final.fit_predict(X_scaled)

print("\n--- K-Means Completed ---")
print("Number of clusters:", selected_k)

# Add cluster labels to customer data

customer_segments = customer_features.copy()

customer_segments["Cluster"] = cluster_labels

print("\n--- Customer Segments ---")
print(
    customer_segments[
        ["CustomerID", "Cluster"]
    ].head(10)
)

# Customers per Cluster

print("\n--- Customers per Cluster ---")

print(
    customer_segments["Cluster"]
    .value_counts()
    .sort_index()
)

# Customers without cluster

print(
    "\nCustomers without cluster:",
    customer_segments["Cluster"].isnull().sum()
)

# final shape

print("\nCustomer Segmentation Shape:")
print(customer_segments.shape)

# Profile Each Cluster

cluster_profile = (
    customer_segments
    .groupby("Cluster")
    [
        [
            "TotalMRR",
            "TotalSeats",
            "SubscriptionCount",
            "TotalLogins",
            "AvgActiveUsers",
            "TotalAPICalls",
            "AvgSessionMinutes",
            "TicketCount",
            "AvgResolutionHours",
            "AvgSatisfactionScore"
        ]
    ]
    .mean()
    .round(2)
)

print("\n--- Cluster Behaviour Profile ---")
print(cluster_profile)

# Revenue and Usage Profile

print("\n--- Revenue and Usage Profile ---")

print(
    cluster_profile[
        [
            "TotalMRR",
            "TotalSeats",
            "TotalLogins",
            "AvgActiveUsers",
            "TotalAPICalls",
            "AvgSessionMinutes"
        ]
    ]
)

print("\n--- Support Profile ---")

print(
    cluster_profile[
        [
            "TicketCount",
            "AvgResolutionHours",
            "AvgSatisfactionScore"
        ]
    ]
)

# cluster sizes

cluster_sizes = (
    customer_segments["Cluster"]
    .value_counts()
    .sort_index()
)

print("\n--- Cluster Sizes ---")
print(cluster_sizes)

# Name Each Cluster by Behaviour

print("\n--- Complete Cluster Profile ---")
print(cluster_profile)

# Highest Value by Cluster

print("\n--- Highest Value by Cluster ---")

for column in [
    "TotalMRR",
    "TotalLogins",
    "AvgActiveUsers",
    "TotalAPICalls",
    "AvgSessionMinutes"
]:
    highest_cluster = cluster_profile[column].idxmax()

    print(
        column,
        "-> Cluster",
        highest_cluster,
        "| Value:",
        cluster_profile.loc[
            highest_cluster, column
        ]
    )
    

# Lowest Value by Cluster

print("\n--- Lowest Value by Cluster ---")

for column in [
    "TotalMRR",
    "TotalLogins",
    "AvgActiveUsers",
    "TotalAPICalls",
    "AvgSessionMinutes"
]:
    lowest_cluster = cluster_profile[column].idxmin()

    print(
        column,
        "-> Cluster",
        lowest_cluster,
        "| Value:",
        cluster_profile.loc[
            lowest_cluster, column
        ]
    )
    
# behavioural names

cluster_names = {
    0: "High-Value Engaged Customers",
    1: "Low-Engagement Customers",
    2: "Support-Heavy Customers",
    3: "Moderate-Usage Customers"
}

# Apply the names

customer_segments["SegmentName"] = (
    customer_segments["Cluster"]
    .map(cluster_names)
)

print("\n--- Customer Segment Names ---")

print(
    customer_segments[
        ["CustomerID", "Cluster", "SegmentName"]
    ].head(15)
)

# Customer Count by Segment

print("\n--- Customer Count by Segment ---")

print(
    customer_segments["SegmentName"]
    .value_counts()
)

# final segment profile

final_segment_profile = (
    customer_segments
    .groupby("SegmentName")
    [
        [
            "TotalMRR",
            "TotalSeats",
            "TotalLogins",
            "AvgActiveUsers",
            "TotalAPICalls",
            "AvgSessionMinutes",
            "TicketCount",
            "AvgResolutionHours",
            "AvgSatisfactionScore"
        ]
    ]
    .mean()
    .round(2)
)

print("\n--- Final Behavioural Segment Profile ---")
print(final_segment_profile)

# Retention Recommendations

print("\n--- Segment Names ---")
print(
    customer_segments["SegmentName"]
    .value_counts()
)

# Retention recommendation for each segment

segment_recommendations = {

    "High-Value Engaged Customers":
        "Use proactive account management, renewal planning, and personalized engagement to maintain strong relationships.",

    "Low-Engagement Customers":
        "Increase product adoption through onboarding refreshers, feature education, and targeted engagement campaigns.",

    "Support-Heavy Customers":
        "Provide proactive support, identify recurring issues, and follow up with customers to improve their experience.",

    "Moderate-Usage Customers":
        "Encourage deeper product adoption through feature recommendations, training, and usage-based engagement."
}

# Apply the recommendations

customer_segments["RetentionRecommendation"] = (
    customer_segments["SegmentName"]
    .map(segment_recommendations)
)

print("\n--- Customer Retention Recommendations ---")

print(
    customer_segments[
        [
            "CustomerID",
            "SegmentName",
            "RetentionRecommendation"
        ]
    ].head(15)
)

# Final Segment Retention Plan

final_retention_plan = (
    customer_segments[
        [
            "SegmentName",
            "RetentionRecommendation"
        ]
    ]
    .drop_duplicates()
    .sort_values("SegmentName")
)

print("\n--- Final Segment Retention Plan ---")
print(final_retention_plan.to_string(index=False))

segment_recommendations = {

    "High Usage - Low Support":
        "Maintain engagement through proactive renewal communication and relevant feature recommendations."
}

# ============================================================
# MODULE 9 - CHURN RISK SCORING
# ============================================================


# Create Customer-Level Features

customer_subscription = (
    subscriptions
    .groupby("CustomerID")
    .agg(
        TotalMRR=("MRR", "sum"),
        TotalSeats=("Seats", "sum")
    )
    .reset_index()
)

print("\n--- Subscription Features ---")
print(customer_subscription.head())

print("\nShape:")
print(customer_subscription.shape)

# usage features:

customer_usage = (
    usage
    .groupby("CustomerID")
    .agg(
        TotalLogins=("Logins", "sum"),
        AvgActiveUsers=("ActiveUsers", "mean"),
        TotalAPICalls=("APICalls", "sum"),
        AvgSessionMinutes=("SessionMinutes", "mean")
    )
    .reset_index()
)

print("\n--- Usage Features ---")
print(customer_usage.head())

print("\nShape:")
print(customer_usage.shape)

# ticket features:

customer_tickets = (
    tickets
    .groupby("CustomerID")
    .agg(
        TicketCount=("TicketID", "nunique"),
        AvgSatisfactionScore=("SatisfactionScore", "mean")
    )
    .reset_index()

)

print("\n--- Ticket Features ---")
print(customer_tickets.head())

print("\nShape:")
print(customer_tickets.shape)

# Combine the Features

# Combine all customer-level features

risk_features = customer_subscription.merge(
    customer_usage,
    on="CustomerID",
    how="outer"
)

risk_features = risk_features.merge(
    customer_tickets,
    on="CustomerID",
    how="outer"
)

print("\n--- Combined Risk Features ---")
print(risk_features.head())

print("\nShape:")
print(risk_features.shape)

print("\nDuplicate CustomerIDs:")
print(
    risk_features["CustomerID"]
    .duplicated()
    .sum()
)

print("\nMissing Values:")
print(risk_features.isnull().sum())

# Handle Missing Values

risk_features["TicketCount"] = (
    risk_features["TicketCount"]
    .fillna(0)
)

risk_features["AvgSatisfactionScore"] = (
    risk_features["AvgSatisfactionScore"]
    .fillna(
        risk_features["AvgSatisfactionScore"].median()
    )
)

usage_columns = [
    "TotalLogins",
    "AvgActiveUsers",
    "TotalAPICalls",
    "AvgSessionMinutes"
]

for column in usage_columns:

    risk_features[column] = (
        risk_features[column]
        .fillna(
            risk_features[column].median()
        )
    )

print("\n--- Missing Values After Cleaning ---")
print(risk_features.isnull().sum())

# Risk Signals

risk_columns = [
    "TotalLogins",
    "AvgActiveUsers",
    "TotalAPICalls",
    "AvgSessionMinutes",
    "TicketCount",
    "AvgSatisfactionScore"
]

print("\n--- Risk Signal Statistics ---")

print(
    risk_features[risk_columns].describe()
)

# Low-Usage Risk Flags

login_threshold = risk_features["TotalLogins"].quantile(0.25)

active_users_threshold = (
    risk_features["AvgActiveUsers"].quantile(0.25)
)

api_threshold = (
    risk_features["TotalAPICalls"].quantile(0.25)
)

session_threshold = (
    risk_features["AvgSessionMinutes"].quantile(0.25)
)

print("\n--- Risk Thresholds ---")

print(
    "Low Login Threshold:",
    round(login_threshold, 2)
)

print(
    "Low Active Users Threshold:",
    round(active_users_threshold, 2)
)

print(
    "Low API Calls Threshold:",
    round(api_threshold, 2)
)

print(
    "Low Session Minutes Threshold:",
    round(session_threshold, 2)
)

# create the flags:

risk_features["LowLoginRisk"] = np.where(
    risk_features["TotalLogins"] <= login_threshold,
    1,
    0
)

risk_features["LowActiveUsersRisk"] = np.where(
    risk_features["AvgActiveUsers"] <= active_users_threshold,
    1,
    0
)

risk_features["LowAPIRisk"] = np.where(
    risk_features["TotalAPICalls"] <= api_threshold,
    1,
    0
)

risk_features["LowSessionRisk"] = np.where(
    risk_features["AvgSessionMinutes"] <= session_threshold,
    1,
    0
)

print("\n--- Usage Risk Flags ---")

print(
    risk_features[
        [
            "CustomerID",
            "LowLoginRisk",
            "LowActiveUsersRisk",
            "LowAPIRisk",
            "LowSessionRisk"
        ]
    ].head(10)
)

# Support Risk Flags

ticket_threshold = (
    risk_features["TicketCount"].quantile(0.75)
)

satisfaction_threshold = (
    risk_features["AvgSatisfactionScore"].quantile(0.25)
)

print("\n--- Support Risk Thresholds ---")

print(
    "High Ticket Threshold:",
    round(ticket_threshold, 2)
)

print(
    "Low Satisfaction Threshold:",
    round(satisfaction_threshold, 2)
)

# Create the flags:

risk_features["HighTicketRisk"] = np.where(
    risk_features["TicketCount"] >= ticket_threshold,
    1,
    0
)

risk_features["LowSatisfactionRisk"] = np.where(
    risk_features["AvgSatisfactionScore"] <= satisfaction_threshold,
    1,
    0
)

print("\n--- Support Risk Flags ---")

print(
    risk_features[
        [
            "CustomerID",
            "TicketCount",
            "AvgSatisfactionScore",
            "HighTicketRisk",
            "LowSatisfactionRisk"
        ]
    ].head(10)
)

# Calculate Risk Score

risk_flag_columns = [
    "LowLoginRisk",
    "LowActiveUsersRisk",
    "LowAPIRisk",
    "LowSessionRisk",
    "HighTicketRisk",
    "LowSatisfactionRisk"
]

risk_features["RiskScore"] = (
    risk_features[risk_flag_columns]
    .sum(axis=1)
)

print("\n--- Risk Score Distribution ---")

print(
    risk_features["RiskScore"]
    .value_counts()
    .sort_index()
)

# Risk Category

risk_features["RiskCategory"] = pd.cut(
    risk_features["RiskScore"],
    bins=[-1, 1, 3, 6],
    labels=[
        "Low Risk",
        "Medium Risk",
        "High Risk"
    ]
)

print("\n--- Risk Category Counts ---")

print(
    risk_features["RiskCategory"]
    .value_counts()
)

# Rank Customers by Risk

risk_features = risk_features.sort_values(
    by=["RiskScore", "TotalMRR"],
    ascending=[False, False]
).reset_index(drop=True)

risk_features["RiskRank"] = (
    risk_features.index + 1
)

print("\n--- Top 20 Customers by Churn Risk ---")

print(
    risk_features[
        [
            "RiskRank",
            "CustomerID",
            "TotalMRR",
            "RiskScore",
            "RiskCategory"
        ]
    ].head(20)
)

# MRR in Highest-Risk Group

high_risk_customers = risk_features[
    risk_features["RiskCategory"] == "High Risk"
].copy()

high_risk_mrr = high_risk_customers["TotalMRR"].sum()

total_mrr = risk_features["TotalMRR"].sum()

high_risk_mrr_percentage = (
    high_risk_mrr / total_mrr * 100
)

print("\n--- High-Risk Customer Analysis ---")

print(
    "High-Risk Customers:",
    len(high_risk_customers)
)

print(
    "High-Risk MRR:",
    round(high_risk_mrr, 2)
)

print(
    "Total MRR:",
    round(total_mrr, 2)
)

print(
    "High-Risk MRR Percentage:",
    round(high_risk_mrr_percentage, 2),
    "%"
)

# Final Risk Table

final_risk_table = risk_features[
    [
        "RiskRank",
        "CustomerID",
        "TotalMRR",
        "TotalLogins",
        "AvgActiveUsers",
        "TotalAPICalls",
        "AvgSessionMinutes",
        "TicketCount",
        "AvgSatisfactionScore",
        "RiskScore",
        "RiskCategory"
    ]
].copy()

print("\n--- Final Churn Risk Table ---")

print(
    final_risk_table.head(20).to_string(index=False)
)

