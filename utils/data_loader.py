from alpha_vantage.fundamentaldata import FundamentalData
import pandas
import json
import requests
import time
import os
API_KEY = "TA3T4E5QIWXC0UBH"
SYMBOL = "NVDA"

#Currently set to 2020 to 2025 quarterly

def get_quarterly_eps():
    url = f"https://www.alphavantage.co/query?function=EARNINGS&symbol={SYMBOL}&apikey={API_KEY}"
    response = requests.get(url)
    data = response.json()

    if "quarterlyEarnings" in data:
        quarterly_earnings = [
            {
                "Date": entry["fiscalDateEnding"],
                "EPS": entry["reportedEPS"],
            }
            for entry in data["quarterlyEarnings"]
            if "2020" <= entry["fiscalDateEnding"][:4] <= "2025"
        ]
        return quarterly_earnings
    else:
        print("Error fetching EPS data:", data)
        return []

def get_quarterly_income_statement():
    url = f"https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol={SYMBOL}&apikey={API_KEY}"
    response = requests.get(url)
    data = response.json()

    if "quarterlyReports" in data:
        quarterly_income = [
            {
                "Date": entry["fiscalDateEnding"],
                "Revenue": entry.get("totalRevenue"),
                "Net Income": entry.get("netIncome"),
                "EPS": entry.get("eps"),
                "Operating Income": entry.get("operatingIncome"),
            }
            for entry in data["quarterlyReports"]
            if "2020" <= entry["fiscalDateEnding"][:4] <= "2025"
        ]
        return quarterly_income
    else:
        print("Error fetching income statement:", data)
        return []

def get_quarterly_balance_sheet():
    url = f"https://www.alphavantage.co/query?function=BALANCE_SHEET&symbol={SYMBOL}&apikey={API_KEY}"
    response = requests.get(url)
    data = response.json()

    if "quarterlyReports" in data:
        quarterly_balance = [
            {
                "Date": entry["fiscalDateEnding"],
                "Total Assets": entry.get("totalAssets"),
                "Total Liabilities": entry.get("totalLiabilities"),
                "Shareholders Equity": entry.get("totalShareholderEquity"),
            }
            for entry in data["quarterlyReports"]
            if "2020" <= entry["fiscalDateEnding"][:4] <= "2025"
        ]
        return quarterly_balance
    else:
        print("Error fetching balance sheet:", data)
        return []

eps_data = get_quarterly_eps()
income_data = get_quarterly_income_statement()

# Wait to avoid hitting API rate limits (5 requests per minute)
time.sleep(12)

balance_data = get_quarterly_balance_sheet()

output_dir = "data/raw/"

os.makedirs(output_dir, exist_ok=True)

print("Saving files in:", os.path.abspath(output_dir))

with open(os.path.join(output_dir, "nvda_quarterly_eps.json"), "w") as file:
    json.dump(eps_data, file, indent=4)
print("Quarterly EPS data saved to nvda_quarterly_eps.json")

with open(os.path.join(output_dir, "nvda_quarterly_income.json"), "w") as file:
    json.dump(income_data, file, indent=4)
print("Quarterly Income Statement data saved to nvda_quarterly_income.json")

with open(os.path.join(output_dir, "nvda_quarterly_balance.json"), "w") as file:
    json.dump(balance_data, file, indent=4)
print("Quarterly Balance Sheet data saved to nvda_quarterly_balance.json")


