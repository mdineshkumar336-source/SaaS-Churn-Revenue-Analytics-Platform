# Week Project – SaaS Churn & Revenue Analytics Platform

# ============================================================
# MODULE 5 — Statistics
# ============================================================

# Imports Libraries and Load Data

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Load cleaned datasets
customers = pd.read_csv("cleaned_customers.csv")
subscriptions = pd.read_csv("cleaned_subscriptions.csv")
usage = pd.read_csv("cleaned_usage.csv")
tickets = pd.read_csv("cleaned_tickets.csv")

print("Customers shape:", customers.shape)
print("Subscriptions shape:", subscriptions.shape)
print("Usage shape:", usage.shape)
print("Tickets shape:", tickets.shape)

# Prepare Statistics Analysis Dataset

# Select required subscription columns
subscription_data = subscriptions[
    ["CustomerID", "SubscriptionID", "Status"]
].copy()

# Select required usage columns
usage_data = usage[
    [
        "CustomerID",
        "SubscriptionID",
        "Logins",
        "ActiveUsers",
        "APICalls",
        "SessionMinutes"
    ]
].copy()

# Merge subscription and usage data
stats_data = subscription_data.merge(
    usage_data,
    on=["CustomerID", "SubscriptionID"],
    how="left"
)

print("\n--- Statistics Dataset ---")
print(stats_data.head())

print("\nShape:", stats_data.shape)

print("\nColumns:")
print(stats_data.columns.tolist())

# Descriptive Statistics

# Key statistical metrics
key_metrics = [
    "Logins",
    "ActiveUsers",
    "APICalls",
    "SessionMinutes"
]

# Check missing values
print("\n--- Missing Values in Key Metrics ---")
print(stats_data[key_metrics].isna().sum())

# Calculate descriptive statistics
print("\n--- Descriptive Statistics ---")
print(
    stats_data[key_metrics].describe().round(2)
)

# Random Sample 1 vs Population

# Remove missing Logins values
logins_population = stats_data["Logins"].dropna()

# Population mean
population_mean = logins_population.mean()

# Take a random sample of 50 observations
sample_1 = logins_population.sample(
    n=50,
    random_state=42
)

# Calculate sample mean
sample_1_mean = sample_1.mean()

print("\n--- Sample 1 vs Population ---")
print("Population size:", len(logins_population))
print("Population mean:", round(population_mean, 2))
print("Sample 1 size:", len(sample_1))
print("Sample 1 mean:", round(sample_1_mean, 2))
print("Difference:", round(sample_1_mean - population_mean, 2))

# Random Sample 2 vs Population

# Take a second random sample of 50 observations
sample_2 = logins_population.sample(
    n=50,
    random_state=100
)

# Calculate sample mean
sample_2_mean = sample_2.mean()

print("\n--- Sample 2 vs Population ---")
print("Population size:", len(logins_population))
print("Population mean:", round(population_mean, 2))
print("Sample 2 size:", len(sample_2))
print("Sample 2 mean:", round(sample_2_mean, 2))
print("Difference:", round(sample_2_mean - population_mean, 2))

# Prepare Churn Hypothesis Groups

# Churned customers
churned_logins = stats_data.loc[
    stats_data["Status"] == "Churned",
    "Logins"
].dropna()

# Active customers = retained group
retained_logins = stats_data.loc[
    stats_data["Status"] == "Active",
    "Logins"
].dropna()

# Display group information
print("\n--- Churn Hypothesis Groups ---")

print("Churned observations:", len(churned_logins))
print(
    "Churned mean Logins:",
    round(churned_logins.mean(), 2)
)

print("\nActive/Retained observations:", len(retained_logins))
print(
    "Active/Retained mean Logins:",
    round(retained_logins.mean(), 2)
)

# Independent T-Test

# Run independent two-sample t-test
t_statistic, p_value_two_sided = stats.ttest_ind(
    churned_logins,
    retained_logins,
    equal_var=False
)

# Convert two-sided p-value to one-sided p-value
# because our hypothesis is specifically:
# Churned customers have LOWER Logins
if churned_logins.mean() < retained_logins.mean():
    p_value = p_value_two_sided / 2
else:
    p_value = 1 - (p_value_two_sided / 2)

print("\n--- Independent T-Test ---")
print("T-statistic:", round(t_statistic, 4))
print("P-value:", round(p_value, 6))

# Interpret the Hypothesis Test

alpha = 0.05

print("\n--- Hypothesis Test Result ---")
print("T-statistic:", round(t_statistic, 4))
print("P-value:", f"{p_value:.10e}")
print("Significance level:", alpha)

if p_value < alpha:
    print("\nDecision: Reject the null hypothesis.")
else:
    print("\nDecision: Fail to reject the null hypothesis.")
    
# Final Statistics Summary

print("\n" + "=" * 60)
print("MODULE 5 - STATISTICS SUMMARY")
print("=" * 60)

# 1. Descriptive statistics
print("\n1. KEY METRICS")
print("Average Logins:", round(stats_data["Logins"].mean(), 2))
print("Average Active Users:", round(stats_data["ActiveUsers"].mean(), 2))
print("Average API Calls:", round(stats_data["APICalls"].mean(), 2))
print(
    "Average Session Minutes:",
    round(stats_data["SessionMinutes"].mean(), 2)
)

# 2. Population vs samples
print("\n2. RANDOM SAMPLE COMPARISON")
print("Population Logins Mean:", round(population_mean, 2))
print("Sample 1 Logins Mean:", round(sample_1_mean, 2))
print("Sample 2 Logins Mean:", round(sample_2_mean, 2))

print(
    "Sample 1 Difference:",
    round(sample_1_mean - population_mean, 2)
)

print(
    "Sample 2 Difference:",
    round(sample_2_mean - population_mean, 2)
)

# 3. Hypothesis test
print("\n3. HYPOTHESIS TEST")
print("Churned Mean Logins:", round(churned_logins.mean(), 2))
print("Active Mean Logins:", round(retained_logins.mean(), 2))
print("T-statistic:", round(t_statistic, 4))
print("P-value:", f"{p_value:.10e}")

# 4. Final decision
if p_value < alpha:
    print("Decision: Reject the null hypothesis.")
    print(
        "Conclusion: Churned customers have significantly "
        "lower login activity than Active customers."
    )
else:
    print("Decision: Fail to reject the null hypothesis.")
    print(
        "Conclusion: There is not enough statistical evidence "
        "to conclude that Churned customers have lower login activity."
    )

# ============================================================
# MODULE 6 — Cohort & Retention Analysis
# ============================================================

# Check Required Columns

print("\n--- Customers Columns ---")
print(customers.columns.tolist())

print("\n--- Subscriptions Columns ---")
print(subscriptions.columns.tolist())

print("\n--- Usage Columns ---")
print(usage.columns.tolist())

# Create Signup Cohort Month

# Convert SignupDate to datetime
customers["SignupDate"] = pd.to_datetime(
    customers["SignupDate"],
    errors="coerce"
)

# Create cohort month
customers["CohortMonth"] = (
    customers["SignupDate"]
    .dt.to_period("M")
)

# Display the result
print("\n--- Customer Cohort Data ---")
print(
    customers[
        ["CustomerID", "SignupDate", "CohortMonth"]
    ].head(10)
)

# Count customers in each cohort
print("\n--- Customers by Cohort ---")
print(
    customers["CohortMonth"]
    .value_counts()
    .sort_index()
)

# Create Activity Month

# Convert usage Month to datetime
usage["Month"] = pd.to_datetime(
    usage["Month"],
    errors="coerce"
)

# Create ActivityMonth
usage["ActivityMonth"] = (
    usage["Month"]
    .dt.to_period("M")
)

# Display the result
print("\n--- Usage Activity Data ---")
print(
    usage[
        [
            "CustomerID",
            "SubscriptionID",
            "Month",
            "ActivityMonth"
        ]
    ].head(10)
)

# Check missing activity months
print("\n--- Missing Activity Months ---")
print(usage["ActivityMonth"].isna().sum())

# Merge Cohort Data with Usage Activity

# Keep only the columns needed from customers
customer_cohort = customers[
    ["CustomerID", "CohortMonth"]
].copy()

# Keep only the columns needed from usage
usage_activity = usage[
    ["CustomerID", "ActivityMonth"]
].copy()

# Remove duplicate customer-month activity
# This prevents one customer from being counted multiple times
# in the same month.
usage_activity = usage_activity.drop_duplicates(
    subset=["CustomerID", "ActivityMonth"]
)

# Merge customer cohort with monthly activity
cohort_activity = customer_cohort.merge(
    usage_activity,
    on="CustomerID",
    how="inner"
)

print("\n--- Cohort Activity Data ---")
print(cohort_activity.head(10))

print("\nShape:", cohort_activity.shape)

print("\nColumns:")
print(cohort_activity.columns.tolist())

# Calculate Cohort Index

cohort_activity["CohortIndex"] = (
    (cohort_activity["ActivityMonth"].dt.year
     - cohort_activity["CohortMonth"].dt.year) * 12
    +
    (cohort_activity["ActivityMonth"].dt.month
     - cohort_activity["CohortMonth"].dt.month)
)

print("\n--- Cohort Activity with Cohort Index ---")
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

print("\n--- Cohort Index Values ---")
print(
    sorted(
        cohort_activity["CohortIndex"]
        .dropna()
        .unique()
    )
)

print("\n--- Negative Cohort Index Records ---")

print(
    cohort_activity[
        cohort_activity["CohortIndex"] < 0
    ]
)

# Count Active Customers by Cohort

# Remove any activity before the customer's signup month
cohort_activity = cohort_activity[
    cohort_activity["CohortIndex"] >= 0
].copy()

# Count unique active customers
cohort_counts = (
    cohort_activity
    .groupby(
        ["CohortMonth", "CohortIndex"]
    )["CustomerID"]
    .nunique()
    .reset_index(name="ActiveCustomers")
)

print("\n--- Active Customers by Cohort ---")
print(cohort_counts.head(20))

# Check the cohort sizes

cohort_sizes = (
    customers
    .groupby("CohortMonth")["CustomerID"]
    .nunique()
)

print("\n--- Customers in Each Cohort ---")
print(cohort_sizes)

# Add Cohort Size to Active Customer Table

# Add original cohort size
cohort_counts["CohortSize"] = (
    cohort_counts["CohortMonth"]
    .map(cohort_sizes)
)

print("\n--- Cohort Counts with Cohort Size ---")
print(cohort_counts.head(20))

# Calculate Retention Rate

cohort_counts["RetentionRate"] = (
    cohort_counts["ActiveCustomers"]
    / cohort_counts["CohortSize"]
)

print("\n--- Retention Rate ---")
print(
    cohort_counts[
        [
            "CohortMonth",
            "CohortIndex",
            "ActiveCustomers",
            "CohortSize",
            "RetentionRate"
        ]
    ].head(20)
)

# Display retention as percentage

cohort_counts["RetentionPercent"] = (
    cohort_counts["RetentionRate"] * 100
)

print("\n--- Retention Percentage ---")
print(
    cohort_counts[
        [
            "CohortMonth",
            "CohortIndex",
            "ActiveCustomers",
            "CohortSize",
            "RetentionPercent"
        ]
    ].head(20)
)

# Month 0 Retention Check

print("\n--- Month 0 Retention Check ---")

print(
    cohort_counts[
        cohort_counts["CohortIndex"] == 0
    ][
        [
            "CohortMonth",
            "ActiveCustomers",
            "CohortSize",
            "RetentionPercent"
        ]
    ]
)

# Create Cohort Retention Table

retention_table = cohort_counts.pivot_table(
    index="CohortMonth",
    columns="CohortIndex",
    values="RetentionRate"
)

print("\n--- Cohort Retention Table ---")
print(retention_table)

# Display as percentages

retention_table_percent = (
    retention_table * 100
).round(1)

print("\n--- Cohort Retention Table (%) ---")
print(retention_table_percent)

# Month 0 Retention

print("\n--- Month 0 Retention ---")
print(retention_table_percent[0])

# Create Retention Curve

# Calculate average retention

retention_curve = retention_table.mean(
    axis=0,
    skipna=True
)

print("\n--- Average Retention by Month ---")
print(
    (retention_curve * 100).round(1)
)

# Plot the Retention Curve

plt.figure(figsize=(10, 6))

plt.plot(
    retention_curve.index,
    retention_curve.values * 100,
    marker="o"
)

plt.xlabel("Months Since Signup")
plt.ylabel("Average Retention (%)")
plt.title("Cohort Retention Curve")

plt.xticks(retention_curve.index)

plt.grid(True)

plt.show()

# Print the values used in the chart

print("\n--- Retention Curve Data ---")

retention_curve_data = pd.DataFrame({
    "MonthSinceSignup": retention_curve.index,
    "AverageRetentionPercent": (
        retention_curve.values * 100
    ).round(2)
})

print(retention_curve_data)

# Identify Best & Worst Cohorts

# Check how many cohorts have data at each month

cohort_availability = (
    retention_table.notna()
    .sum(axis=0)
)

print("\n--- Number of Cohorts Available at Each Month ---")
print(cohort_availability)

# Select a Comparable Month

candidate_months = [3, 2, 1]

comparison_month = None

for month in candidate_months:
    if (
        month in retention_table.columns
        and retention_table[month].notna().sum() >= 2
    ):
        comparison_month = month
        break

if comparison_month is None:
    print("Not enough cohorts available for comparison.")
else:
    print(
        f"\nComparison Month: Month {comparison_month}"
    )
    
# Compare the Cohorts

if comparison_month is not None:

    cohort_comparison = (
        retention_table[
            [comparison_month]
        ]
        .dropna()
        .sort_values(
            by=comparison_month,
            ascending=False
        )
    )

    print(
        f"\n--- Cohort Retention at Month {comparison_month} ---"
    )

    print(
        (
            cohort_comparison * 100
        ).round(1)
    )
    
# Identify Best and Worst

if comparison_month is not None:

    best_cohort = cohort_comparison[
        comparison_month
    ].idxmax()

    worst_cohort = cohort_comparison[
        comparison_month
    ].idxmin()

    best_retention = (
        cohort_comparison.loc[
            best_cohort,
            comparison_month
        ] * 100
    )

    worst_retention = (
        cohort_comparison.loc[
            worst_cohort,
            comparison_month
        ] * 100
    )

    print(
        f"\nBest-performing cohort at Month {comparison_month}: "
        f"{best_cohort}"
    )

    print(
        f"Retention: {best_retention:.1f}%"
    )

    print(
        f"\nWorst-performing cohort at Month {comparison_month}: "
        f"{worst_cohort}"
    )

    print(
        f"Retention: {worst_retention:.1f}%"
    )
    
# Find What Changed

if comparison_month is not None:

    retention_difference = (
        best_retention - worst_retention
    )

    print(
        f"\nRetention difference: "
        f"{retention_difference:.1f} percentage points"
    )
    
# Final Cohort Retention Insights

# Print the retention curve summary

print("\n" + "=" * 60)
print("MODULE 6 - COHORT & RETENTION ANALYSIS SUMMARY")
print("=" * 60)

print("\nAverage Retention by Month:")
print(retention_curve_data)

# Print best and worst cohort

if comparison_month is not None:

    print(
        f"\nComparison point: Month {comparison_month}"
    )

    print(
        f"Best-performing cohort: {best_cohort}"
    )

    print(
        f"Best retention: {best_retention:.1f}%"
    )

    print(
        f"Worst-performing cohort: {worst_cohort}"
    )

    print(
        f"Worst retention: {worst_retention:.1f}%"
    )

    print(
        f"Difference: {retention_difference:.1f} "
        f"percentage points"
    )
    
# Overall Retention Change

valid_curve = retention_curve.dropna()

first_month = valid_curve.index.min()
last_month = valid_curve.index.max()

first_retention = (
    valid_curve.loc[first_month] * 100
)

last_retention = (
    valid_curve.loc[last_month] * 100
)

retention_change = (
    last_retention - first_retention
)

print(
    f"\nRetention at Month {first_month}: "
    f"{first_retention:.1f}%"
)

print(
    f"Retention at Month {last_month}: "
    f"{last_retention:.1f}%"
)

print(
    f"Overall change: "
    f"{retention_change:.1f} percentage points"
)

# Final Insight

print("\n--- Final Insight ---")

if retention_change < 0:
    print(
        "Retention decreased as the number of months since signup increased."
    )
elif retention_change > 0:
    print(
        "Retention increased over the observed period."
    )
else:
    print(
        "Retention remained stable over the observed period."
    )

if comparison_month is not None:
    print(
        f"At Month {comparison_month}, "
        f"{best_cohort} had the highest observed retention "
        f"and {worst_cohort} had the lowest observed retention."
    )

print(
    "Newer cohorts have shorter observation periods, "
    "so later-month retention should be interpreted carefully."
)
