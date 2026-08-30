
import os
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv
from plyer import notification
from langchain_google_genai import ChatGoogleGenerativeAI


# Load the Gemini API key from .env
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in the .env file.")

os.environ["GOOGLE_API_KEY"] = api_key

print("Gemini API key loaded successfully.")


# Load customer usage data
csv_file = "data/usage_logs.csv"

if not os.path.exists(csv_file):
    raise FileNotFoundError(f"Could not find the CSV file: {csv_file}")

customers = pd.read_csv(csv_file)

print(f"Loaded {len(customers)} customer records.")


# Check that the CSV has the required columns
required_columns = [
    "customer_id",
    "customer_name",
    "last_used_date"
]

missing_columns = [
    column for column in required_columns
    if column not in customers.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns in CSV: {', '.join(missing_columns)}"
    )


# Calculate how many days each customer has been inactive
today = datetime.today()

customers["last_used_date"] = pd.to_datetime(
    customers["last_used_date"],
    errors="coerce"
)

customers["inactive_days"] = (
    today - customers["last_used_date"]
).dt.days


# Find customers who have been inactive for more than 30 days
inactive_customers = customers[
    customers["inactive_days"] > 30
].copy()

print(
    f"Customers inactive for more than 30 days: "
    f"{len(inactive_customers)}"
)


# If there are no inactive customers, stop here
if inactive_customers.empty:
    print("\n================================")
    print("       CUSTOMER CHURN ALERT")
    print("================================\n")

    print("No customers are at risk of churn.")

    notification.notify(
        title="Customer Churn Alert",
        message="No customers have been inactive for more than 30 days.",
        timeout=10
    )

    exit()


# Prepare the inactive customer information for Gemini
customer_details = []

for _, customer in inactive_customers.iterrows():
    customer_details.append(
        f"""
Customer ID: {customer['customer_id']}
Customer Name: {customer['customer_name']}
Last Active Date: {customer['last_used_date'].strftime('%Y-%m-%d')}
Inactive Days: {customer['inactive_days']}
"""
    )

customer_data = "\n".join(customer_details)


# Create the Gemini AI model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)


# Ask Gemini to analyze the customers
prompt = f"""
You are a Customer Churn Prediction AI.

Today's date is {today.strftime('%Y-%m-%d')}.

These customers have been inactive for more than 30 days:

{customer_data}

Analyze each customer and provide:

- Customer Name
- Customer ID
- Last Active Date
- Inactive Days
- Risk Level
- Reason
- Recommended Action

Use these risk levels:

HIGH:
More than 60 days inactive.

MEDIUM:
31 to 60 days inactive.

At the end, create a simple alert for each customer.

Example:

ALERT:
Customer John has been inactive for 45 days.
Risk Level: MEDIUM.
Recommended Action: Contact the customer immediately.

Only include the customers provided above.
"""


try:
    response = llm.invoke(prompt)
    result = response.content

except Exception as error:
    print("\nGemini API Error:")
    print(error)

    print(
        "\nCustomer inactivity was calculated successfully, "
        "but Gemini could not generate the analysis."
    )

    exit()


# Display the churn analysis
print("\n========================================")
print("         CUSTOMER CHURN ALERT")
print("========================================\n")

print(result)


# Send a Windows desktop notification
notification.notify(
    title="Customer Churn Alert",
    message=(
        f"{len(inactive_customers)} customer(s) "
        "have been inactive for more than 30 days."
    ),
    timeout=10
)

print("\nDesktop notification sent successfully.")
