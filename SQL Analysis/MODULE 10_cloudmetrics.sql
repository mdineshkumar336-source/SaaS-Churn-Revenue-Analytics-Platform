-- Week Project – SaaS Churn & Revenue Analytics Platform

-- Create Database

CREATE DATABASE cloudmetrics;
USE cloudmetrics;

SELECT DATABASE();

-- Create Customers Table

CREATE TABLE customers (
    CustomerID VARCHAR(20) PRIMARY KEY,
    CompanyName VARCHAR(255),
    Industry VARCHAR(100),
    Country VARCHAR(100),
    City VARCHAR(100),
    EmployeeCount INT,
    SignupDate DATE,
    AcquisitionChannel VARCHAR(100)
);

DESCRIBE customers;

SELECT COUNT(*) AS CustomerRows
FROM customers;

SELECT *
FROM customers
LIMIT 5;

-- Create Subscriptions Table

CREATE TABLE subscriptions (
    SubscriptionID VARCHAR(20) PRIMARY KEY,
    CustomerID VARCHAR(20),
    PlanName VARCHAR(100),
    BillingTerm VARCHAR(50),
    Seats INT,
    MRR DECIMAL(12,2),
    StartDate DATE,
    EndDate DATE,
    Status VARCHAR(50)
);

DESCRIBE subscriptions;

SELECT COUNT(*) AS SubscriptionRows
FROM subscriptions;

SELECT *
FROM subscriptions
LIMIT 5;

SELECT DISTINCT Status
FROM subscriptions;

SELECT DISTINCT PlanName
FROM subscriptions;

-- Create Usage Table

CREATE TABLE usage_data (
    CustomerID VARCHAR(20),
    SubscriptionID VARCHAR(20),
    Month DATE,
    Logins INT,
    ActiveUsers INT,
    FeatureUsed VARCHAR(255),
    APICalls INT,
    SessionMinutes DECIMAL(12,2)
);

DESCRIBE usage_data;

SELECT COUNT(*) AS UsageRows
FROM usage_data;

SELECT *
FROM usage_data
LIMIT 5;

-- Create Tickets Table

CREATE TABLE tickets (
    TicketID VARCHAR(20) PRIMARY KEY,
    CustomerID VARCHAR(20),
    OpenedDate DATE,
    Category VARCHAR(100),
    Priority VARCHAR(50),
    ResolutionHours DECIMAL(12,2),
    SatisfactionScore DECIMAL(5,2)
);

DESCRIBE tickets;

SELECT COUNT(*) AS TicketRows
FROM tickets;

SELECT *
FROM tickets
LIMIT 5;

-- JOIN

SELECT
    c.CustomerID,
    c.CompanyName,
    s.SubscriptionID,
    s.PlanName,
    s.MRR,
    s.Status
FROM customers c
INNER JOIN subscriptions s
    ON c.CustomerID = s.CustomerID;

-- GROUP BY

SELECT
    PlanName,
    COUNT(DISTINCT CustomerID) AS CustomerCount,
    SUM(MRR) AS TotalMRR,
    AVG(MRR) AS AverageMRR
FROM subscriptions
GROUP BY PlanName
ORDER BY TotalMRR DESC;

-- GROUP BY + HAVING

SELECT
    Industry,
    COUNT(*) AS CustomerCount
FROM customers
GROUP BY Industry
HAVING COUNT(*) > 10
ORDER BY CustomerCount DESC;    

-- CASE

SELECT
    CustomerID,
    SubscriptionID,
    PlanName,
    MRR,
    CASE
        WHEN MRR >= 1000 THEN 'High Revenue'
        WHEN MRR >= 500 THEN 'Medium Revenue'
        ELSE 'Low Revenue'
    END AS RevenueCategory
FROM subscriptions;

-- SUBQUERY

SELECT
    SubscriptionID,
    CustomerID,
    PlanName,
    MRR
FROM subscriptions
WHERE MRR > (
    SELECT AVG(MRR)
    FROM subscriptions
)
ORDER BY MRR DESC;

-- CTE

WITH customer_revenue AS (

    SELECT
        CustomerID,
        SUM(MRR) AS TotalMRR
    FROM subscriptions
    GROUP BY CustomerID

)

SELECT
    CustomerID,
    TotalMRR
FROM customer_revenue
WHERE TotalMRR > 1000
ORDER BY TotalMRR DESC;

-- WINDOW FUNCTION

SELECT
    SubscriptionID,
    CustomerID,
    PlanName,
    MRR,
    RANK() OVER (
        PARTITION BY PlanName
        ORDER BY MRR DESC
    ) AS MRRRank
FROM subscriptions;

-- ORPHAN CustomerID

SELECT
    s.SubscriptionID,
    s.CustomerID,
    s.PlanName,
    s.MRR
FROM subscriptions s
LEFT JOIN customers c
    ON s.CustomerID = c.CustomerID
WHERE c.CustomerID IS NULL;

