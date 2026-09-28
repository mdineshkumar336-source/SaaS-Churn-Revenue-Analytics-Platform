# Week Project – SaaS Churn & Revenue Analytics Platform

# ============================================================
# MODULE 3 — NumPy
# ============================================================

# Imports

import pandas as pd
import numpy as np
import os

# Cleaned File Loader

customers = pd.read_csv("cleaned_customers.csv")
subscriptions = pd.read_csv("cleaned_subscriptions.csv")
usage = pd.read_csv("cleaned_usage.csv")
tickets = pd.read_csv("cleaned_tickets.csv")

print("Customers:", customers.shape)
print("Subscriptions:", subscriptions.shape)
print("Usage:", usage.shape)
print("Tickets:", tickets.shape)

# Check required columns

print("\nCustomers columns:")
print(customers.columns.tolist())

print("\nSubscriptions columns:")
print(subscriptions.columns.tolist())

print("\nUsage columns:")
print(usage.columns.tolist())

print("\nTickets columns:")
print(tickets.columns.tolist())

# Convert MRR and Seats to NumPy arrays

mrr_array = subscriptions["MRR"].to_numpy()
seats_array = subscriptions["Seats"].to_numpy()

print("\nMRR NumPy array:")
print(mrr_array)

print("\nSeats NumPy array:")
print(seats_array)

print("\nMRR type:", type(mrr_array))
print("Seats type:", type(seats_array))

# Convert ALL usage metrics to NumPy arrays

logins_array = usage["Logins"].to_numpy()
active_users_array = usage["ActiveUsers"].to_numpy()
feature_used_array = usage["FeatureUsed"].to_numpy()
api_calls_array = usage["APICalls"].to_numpy()
session_minutes_array = usage["SessionMinutes"].to_numpy()

print("\nUsage metrics converted to NumPy arrays")

print("Logins type:", type(logins_array))
print("ActiveUsers type:", type(active_users_array))
print("FeatureUsed type:", type(feature_used_array))
print("APICalls type:", type(api_calls_array))
print("SessionMinutes type:", type(session_minutes_array))

# Mean, Std, Min, Max

# MRR

print("\n--- MRR Statistics ---")

print("Mean:", np.mean(mrr_array))
print("Std:", np.std(mrr_array))
print("Min:", np.min(mrr_array))
print("Max:", np.max(mrr_array))

# Seats

print("\n--- Seats Statistics ---")

print("Mean:", np.mean(seats_array))
print("Std:", np.std(seats_array))
print("Min:", np.min(seats_array))
print("Max:", np.max(seats_array))

# Usage Metrics

print("\n--- Usage Statistics ---")

# Logins
print("\nLogins")
print("Mean:", np.mean(logins_array))
print("Std:", np.std(logins_array))
print("Min:", np.min(logins_array))
print("Max:", np.max(logins_array))

# Active Users
print("\nActiveUsers")
print("Mean:", np.mean(active_users_array))
print("Std:", np.std(active_users_array))
print("Min:", np.min(active_users_array))
print("Max:", np.max(active_users_array))

# API Calls
print("\nAPICalls")
print("Mean:", np.mean(api_calls_array))
print("Std:", np.std(api_calls_array))
print("Min:", np.min(api_calls_array))
print("Max:", np.max(api_calls_array))

# Session Minutes
print("\nSessionMinutes")
print("Mean:", np.mean(session_minutes_array))
print("Std:", np.std(session_minutes_array))
print("Min:", np.min(session_minutes_array))
print("Max:", np.max(session_minutes_array))

# Normalize MRR

mrr_min = np.min(mrr_array)
mrr_max = np.max(mrr_array)

mrr_normalized = (
    (mrr_array - mrr_min) /
    (mrr_max - mrr_min)
)

print("\n--- Normalized MRR ---")
print(mrr_normalized)

print("Normalized MRR Min:", np.min(mrr_normalized))
print("Normalized MRR Max:", np.max(mrr_normalized))

# High-Value Accounts using np.where()

average_mrr = np.mean(mrr_array)

high_value_flag = np.where(
    mrr_array > average_mrr,
    "High Value",
    "Normal"
)

print("\n--- High-Value Account Flag ---")
print(high_value_flag)

# At-Risk Accounts using np.where()

average_session = np.mean(session_minutes_array)

at_risk_flag = np.where(
    session_minutes_array < average_session,
    "At Risk",
    "Normal"
)

print("\n--- At-Risk Account Flag ---")
print(at_risk_flag)

# ============================================================
# MODULE 4 — Pandas Wrangling & EDA
# ============================================================

# Check columns

print("\nCustomers columns:")
print(customers.columns.tolist())

print("\nSubscriptions columns:")
print(subscriptions.columns.tolist())

print("\nUsage columns:")
print(usage.columns.tolist())

print("\nTickets columns:")
print(tickets.columns.tolist())

# Merge all four tables

# Customers + Subscriptions

customer_view = customers.merge(
    subscriptions,
    on="CustomerID",
    how="left",
    suffixes=("_customer", "_subscription")
)

print("\nAfter Customers + Subscriptions merge:")
print(customer_view.shape)

# Add Usage

customer_view = customer_view.merge(
    usage,
    on=["CustomerID", "SubscriptionID"],
    how="left"
)

print("\nAfter adding Usage:")
print(customer_view.shape)

# Add Tickets

customer_view = customer_view.merge(
    tickets,
    on="CustomerID",
    how="left",
    suffixes=("", "_ticket")
)

print("\nFinal customer-level view:")
print(customer_view.shape)

# Check the merged data

print("\nFirst 5 rows:")
print(customer_view.head())

print("\nMissing values after merge:")
print(customer_view.isna().sum())

# GroupBy with multiple aggregations

# Basic Structure

# 1. Plan Summary
plan_summary = customer_view.groupby("PlanName").agg(
    Customers=("CustomerID", "nunique"),
    Total_MRR=("MRR", "sum"),
    Average_MRR=("MRR", "mean"),
    Total_Seats=("Seats", "sum")
).reset_index()

print("\n--- Plan Summary ---")
print(plan_summary)


# 2. Industry Summary
industry_summary = customer_view.groupby("Industry").agg(
    Customers=("CustomerID", "nunique"),
    Total_MRR=("MRR", "sum"),
    Average_MRR=("MRR", "mean"),
    Total_Seats=("Seats", "sum")
).reset_index()

print("\n--- Industry Summary ---")
print(industry_summary)


# 3. Acquisition Channel Summary
channel_summary = customer_view.groupby("AcquisitionChannel").agg(
    Customers=("CustomerID", "nunique"),
    Total_MRR=("MRR", "sum"),
    Average_MRR=("MRR", "mean"),
    Total_Seats=("Seats", "sum")
).reset_index()

print("\n--- Acquisition Channel Summary ---")
print(channel_summary)

# Create Pivot Tables

# Pivot 1 - Plan vs Acquisition Channel
plan_channel_pivot = pd.pivot_table(
    customer_view,
    index="PlanName",
    columns="AcquisitionChannel",
    values="MRR",
    aggfunc="sum",
    fill_value=0
)

print("\n--- Plan vs Acquisition Channel MRR Pivot ---")
print(plan_channel_pivot)


# Pivot 2 - Industry vs Acquisition Channel
industry_channel_pivot = pd.pivot_table(
    customer_view,
    index="Industry",
    columns="AcquisitionChannel",
    values="MRR",
    aggfunc="sum",
    fill_value=0
)

print("\n--- Industry vs Acquisition Channel MRR Pivot ---")
print(industry_channel_pivot)

# CALCULATE CUSTOMER TENURE

# Convert SignupDate into datetime format
customer_view["SignupDate"] = pd.to_datetime(
    customer_view["SignupDate"],
    errors="coerce"
)

# Get today's date
today = pd.Timestamp.today()

# Calculate tenure in months
customer_view["TenureMonths"] = (
    (today - customer_view["SignupDate"]).dt.days / 30.44
).round(1)

# Display the result
print("\n--- Customer Tenure ---")
print(
    customer_view[
        ["CustomerID", "SignupDate", "TenureMonths"]
    ].head(10)
)

# CALCULATE REVENUE PER SEAT

customer_view["RevenuePerSeat"] = np.where(
    customer_view["Seats"] > 0,
    customer_view["MRR"] / customer_view["Seats"],
    np.nan
)

# Display the result
print("\n--- Revenue Per Seat ---")
print(
    customer_view[
        ["CustomerID", "MRR", "Seats", "RevenuePerSeat"]
    ].head(10)
)

# CALCULATE TICKETS PER MONTH

# Convert OpenedDate to datetime
tickets["OpenedDate"] = pd.to_datetime(
    tickets["OpenedDate"],
    errors="coerce"
)

# Create ticket month
tickets["TicketMonth"] = tickets["OpenedDate"].dt.to_period("M")

# Count tickets for each customer in each month
tickets_per_month = (
    tickets
    .groupby(["CustomerID", "TicketMonth"])
    .agg(
        TicketsPerMonth=("TicketID", "count")
    )
    .reset_index()
)

print("\n--- Tickets Per Month ---")
print(tickets_per_month.head(10))

# CALCULATE USAGE TREND

# Convert Month to datetime
usage["Month"] = pd.to_datetime(
    usage["Month"],
    errors="coerce"
)

# Sort by customer and month
usage = usage.sort_values(
    ["CustomerID", "Month"]
)

# Get previous month's SessionMinutes
usage["PreviousSessionMinutes"] = (
    usage.groupby("CustomerID")["SessionMinutes"]
    .shift(1)
)

# Compare current month with previous month
usage["UsageTrend"] = np.where(
    usage["PreviousSessionMinutes"].isna(),
    "No Previous Month",
    np.where(
        usage["SessionMinutes"] > usage["PreviousSessionMinutes"],
        "Increasing",
        np.where(
            usage["SessionMinutes"] < usage["PreviousSessionMinutes"],
            "Decreasing",
            "Stable"
        )
    )
)

# Display the result
print("\n--- Usage Trend ---")
print(
    usage[
        [
            "CustomerID",
            "Month",
            "SessionMinutes",
            "PreviousSessionMinutes",
            "UsageTrend"
        ]
    ].head(15)
)

# DETECT MRR OUTLIERS USING IQR

# Calculate Q1 and Q3
Q1 = customer_view["MRR"].quantile(0.25)
Q3 = customer_view["MRR"].quantile(0.75)

# Calculate IQR
IQR = Q3 - Q1

# Calculate lower and upper bounds
lower_bound = Q1 - (1.5 * IQR)
upper_bound = Q3 + (1.5 * IQR)

# Create outlier flag
customer_view["MRR_Outlier"] = np.where(
    (customer_view["MRR"] < lower_bound) |
    (customer_view["MRR"] > upper_bound),
    "Outlier",
    "Normal"
)

# Display the IQR values
print("\n--- MRR Outlier Detection ---")
print("Q1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)
print("Lower Bound:", lower_bound)
print("Upper Bound:", upper_bound)

# Count normal and outlier records
print("\nOutlier Count:")
print(customer_view["MRR_Outlier"].value_counts())

# Display detected outliers
print("\n--- MRR Outliers ---")
print(
    customer_view[
        customer_view["MRR_Outlier"] == "Outlier"
    ][
        ["CustomerID", "MRR", "MRR_Outlier"]
    ]
)

# CORRELATION MATRIX

# Select numeric columns
numeric_columns = [
    "MRR",
    "Seats",
    "Logins",
    "ActiveUsers",
    "APICalls",
    "SessionMinutes",
    "TenureMonths",
    "RevenuePerSeat"
]

# Calculate correlation
correlation_matrix = customer_view[numeric_columns].corr()

print("\n--- Correlation Matrix ---")
print(correlation_matrix.round(2))

