# Week Project – SaaS Churn & Revenue Analytics Platform

# ============================================================
# MODULE 1 — Python Foundations
# ============================================================

# Imports

import pandas as pd
import numpy as np
import os

# File Loader

def load_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded successfully: {file_path}")
        return df

    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None

    except pd.errors.ParserError:
        print(f"File is malformed: {file_path}")
        return None

    except Exception as e:
        print(f"Unexpected error: {e}")
        return None
    
customers = load_csv("saas_customers.csv")
subscriptions = load_csv("saas_subscriptions.csv")
usage = load_csv("saas_usage.csv")
tickets = load_csv("saas_tickets.csv")

# Validation Functions

def validate_dataframe(df, table_name):
    print("\n" + "=" * 50)
    print(f"VALIDATION: {table_name}")
    print("=" * 50)

    print("Shape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())

    print("\nData Types:")
    print(df.dtypes)

    print("\nMissing Values:")
    print(df.isna().sum())

    print("\nDuplicate Rows:", df.duplicated().sum())
    
validate_dataframe(customers, "Customers")
validate_dataframe(subscriptions, "Subscriptions")
validate_dataframe(usage, "Usage")
validate_dataframe(tickets, "Tickets")

# ============================================================
# MODULE 2 — Data Audit & Cleaning
# ============================================================

# Audit Function

def audit_dataframe(df, table_name):

    print("\n" + "=" * 60)
    print(f"AUDIT REPORT: {table_name}")
    print("=" * 60)

    print("\nShape:")
    print(df.shape)

    print("\nData Types:")
    print(df.dtypes)

    print("\nMissing Values:")
    print(df.isna().sum())

    print("\nDuplicate Rows:")
    print(df.duplicated().sum())

    print("\nUnique Values for Text Columns:")

    text_columns = df.select_dtypes(include="object").columns

    for col in text_columns:
        print(f"\n{col}:")
        print(df[col].dropna().unique())
        
audit_dataframe(customers, "Customers")
audit_dataframe(subscriptions, "Subscriptions")
audit_dataframe(usage, "Usage")
audit_dataframe(tickets, "Tickets")

# Standardise Text

def standardize_text(df):

    df = df.copy()

    text_columns = df.select_dtypes(include="object").columns

    for col in text_columns:
        df[col] = df[col].astype("string").str.strip()

    return df

customers = standardize_text(customers)
subscriptions = standardize_text(subscriptions)
usage = standardize_text(usage)
tickets = standardize_text(tickets)

# Standardise Categories

# Customers:

customers["Industry"] = customers["Industry"].str.title()
customers["Country"] = customers["Country"].str.title()
customers["City"] = customers["City"].str.title()
customers["AcquisitionChannel"] = customers["AcquisitionChannel"].str.title()

# Subscriptions:

subscriptions["PlanName"] = subscriptions["PlanName"].str.title()
subscriptions["BillingTerm"] = subscriptions["BillingTerm"].str.title()
subscriptions["Status"] = subscriptions["Status"].str.title()

# Tickets:

tickets["Category"] = tickets["Category"].str.title()
tickets["Priority"] = tickets["Priority"].str.title()

# Usage:

usage["FeatureUsed"] = usage["FeatureUsed"].str.title()

# Handle Customers missing values

customers["Industry"] = customers["Industry"].fillna("Unknown")

# Employee Count

employee_median = customers["EmployeeCount"].median()

customers["EmployeeCount"] = customers["EmployeeCount"].fillna(
    employee_median
)

# Acquisition Channel

customers["AcquisitionChannel"] = (
    customers["AcquisitionChannel"].fillna("Unknown")
)

# Subscriptions missing values

# Seats

subscriptions["Seats"] = subscriptions["Seats"].fillna(
    subscriptions["Seats"].median()
)

# MRR

subscriptions["MRR"] = subscriptions["MRR"].fillna(
    subscriptions["MRR"].median()
)

# EndDate is intentionally left as NaN for ongoing subscriptions.

# Remove exact duplicates

customer_duplicates = customers.duplicated().sum()
subscription_duplicates = subscriptions.duplicated().sum()
usage_duplicates = usage.duplicated().sum()
ticket_duplicates = tickets.duplicated().sum()

print("Customers duplicate rows =", customer_duplicates)
print("Subscriptions duplicate rows =", subscription_duplicates)
print("Usage duplicate rows =", usage_duplicates)
print("Tickets duplicate rows =", ticket_duplicates)

# Remove

customers = customers.drop_duplicates()
subscriptions = subscriptions.drop_duplicates()
usage = usage.drop_duplicates()
tickets = tickets.drop_duplicates()

print("Customers duplicates after cleaning:", customers.duplicated().sum())
print("Subscriptions duplicates after cleaning:", subscriptions.duplicated().sum())
print("Usage duplicates after cleaning:", usage.duplicated().sum())
print("Tickets duplicates after cleaning:", tickets.duplicated().sum())

# Date conversion

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

# Referential integrity

orphan_subscriptions = subscriptions[
    ~subscriptions["CustomerID"].isin(customers["CustomerID"])
]

print("Orphan subscription records:")
print(orphan_subscriptions)

# Usage

orphan_usage = usage[
    ~usage["CustomerID"].isin(customers["CustomerID"])
]

print("Orphan usage records:")
print(orphan_usage)

# Tickets

orphan_tickets = tickets[
    ~tickets["CustomerID"].isin(customers["CustomerID"])
]

print("Orphan ticket records:")
print(orphan_tickets)

# Save cleaned datasets

customers.to_csv("cleaned_customers.csv", index=False)
subscriptions.to_csv("cleaned_subscriptions.csv", index=False)
usage.to_csv("cleaned_usage.csv", index=False)
tickets.to_csv("cleaned_tickets.csv", index=False)

print("Cleaned files saved successfully!")