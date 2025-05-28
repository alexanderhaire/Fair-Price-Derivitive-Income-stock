import yfinance as yf
import numpy as np
import pandas as pd
import datetime

# Default parameters based on prospectus assumptions
REQUIRED_YIELD = 0.15        # 15% annual required return (reflects high risk)
DEFAULT_TREASURY_YIELD = 0.045  # Fallback Treasury yield (4.5%)
DEFAULT_EXPENSE_RATIO = 0.0099  # 0.99% annual expenses fallback
CALL_STRIKE_MULTIPLIER = 1.10   # 10% out-of-the-money strike price
NAV_EROSION_FACTOR = 0.07       # 7% expected NAV erosion due to return of capital (ROC)

# Additional parameters for option income adjustments
OPTION_RISK_PREMIUM = 0.03      # 3% risk premium factor (reflecting capped upside risk)
DEFAULT_VOLATILITY = 0.50       # Default annualized volatility for COIN (50%)

def get_latest_price(ticker):
    """
    Fetch the latest closing price for a given ticker.
    """
    data = yf.Ticker(ticker).history(period="1d")
    if data.empty:
        raise ValueError(f"No price data found for ticker: {ticker}")
    return data['Close'].iloc[-1]

def get_treasury_yield():
    """
    Fetch the latest short-term U.S. Treasury yield.
    Attempts to use the ^IRX ticker (13-week T-bill yield) from Yahoo Finance.
    If unsuccessful, returns a default yield.
    """
    try:
        treasury_data = yf.Ticker("^IRX").history(period="1d")
        # ^IRX is quoted in percentage points (e.g., 4.5 means 4.5%)
        if treasury_data.empty:
            return DEFAULT_TREASURY_YIELD
        yield_percent = treasury_data['Close'].iloc[-1] / 100
        return yield_percent
    except Exception as e:
        print(f"Error fetching Treasury yield: {e}")
        return DEFAULT_TREASURY_YIELD

def get_expense_ratio(ticker):
    """
    Attempt to fetch the expense ratio from the ticker's info.
    If not available, return the default expense ratio.
    """
    try:
        info = yf.Ticker(ticker).info
        expense = info.get('expenseRatio', DEFAULT_EXPENSE_RATIO)
        # Sometimes expenseRatio might be in decimal form already.
        return expense if expense else DEFAULT_EXPENSE_RATIO
    except Exception as e:
        print(f"Error fetching expense ratio for {ticker}: {e}")
        return DEFAULT_EXPENSE_RATIO

def get_call_option_premium(ticker, strike_multiplier):
    """
    Fetch the call option premium for the nearest expiration date.
    The function selects the call option whose strike is closest to current price * multiplier.
    """
    ticker_obj = yf.Ticker(ticker)
    exp_dates = ticker_obj.options
    if not exp_dates:
        raise ValueError(f"No options available for ticker: {ticker}")
    closest_exp = exp_dates[0]  # Use the nearest expiration date
    option_chain = ticker_obj.option_chain(closest_exp)
    calls = option_chain.calls

    current_price = get_latest_price(ticker)
    target_strike = current_price * strike_multiplier
    closest_call = calls.iloc[(calls['strike'] - target_strike).abs().idxmin()]

    return closest_call['lastPrice']

def get_historical_volatility(ticker, period="1y"):
    """
    Calculate the annualized historical volatility of a ticker.
    Retrieves daily closing prices for the specified period,
    computes daily log returns, then annualizes the standard deviation.
    """
    data = yf.Ticker(ticker).history(period=period)
    if data.empty:
        return DEFAULT_VOLATILITY
    prices = data['Close']
    log_returns = np.log(prices / prices.shift(1)).dropna()
    daily_vol = np.std(log_returns)
    annual_vol = daily_vol * np.sqrt(252)  # Assuming 252 trading days per year
    return annual_vol

def calculate_adjusted_option_income(call_option_premium, volatility, risk_premium):
    """
    Calculate the adjusted annual option income.
    Annualizes the monthly call option premium and then adjusts it downward
    by a factor proportional to the product of the risk premium and volatility.
    
    This simplified adjustment reflects that higher volatility (and risk)
    reduces the effective income because of the capped upside.
    """
    base_income = call_option_premium * 12  # Annualize the premium
    adjusted_income = base_income * (1 - risk_premium * volatility)
    return adjusted_income

def compute_cony_fair_value():
    """
    Computes the fair value of the CONY ETF using a dividend discount model (DDM)
    adjusted for NAV erosion, option strategy risks, and expense drag.
    
    Income Sources:
      - Adjusted option income (based on COIN's call premium and historical volatility)
      - Treasury income (assumed on 50% of NAV using current Treasury yield)
    
    The final fair value is the lower of the DDM valuation and the NAV adjusted for return of capital erosion.
    """
    # Fetch dynamic data
    cony_nav = get_latest_price("CONY")
    coin_price = get_latest_price("COIN")
    expense_ratio = get_expense_ratio("CONY")
    treasury_yield = get_treasury_yield()
    
    # Retrieve COIN option premium and historical volatility
    call_premium = get_call_option_premium("COIN", CALL_STRIKE_MULTIPLIER)
    volatility = get_historical_volatility("COIN", period="1y")
    
    # Calculate adjusted annual option income
    annual_option_income = calculate_adjusted_option_income(call_premium, volatility, OPTION_RISK_PREMIUM)
    
    # Treasury income is assumed on 50% of CONY's NAV at the current Treasury yield
    annual_treasury_income = (cony_nav * 0.5) * treasury_yield
    
    # Total annual gross income
    gross_income = annual_option_income + annual_treasury_income
    
    # Annual expenses based on the fetched expense ratio
    annual_expenses = cony_nav * expense_ratio
    
    # Net distributable income after expenses
    net_distribution = gross_income - annual_expenses
    
    # Fair value using a Dividend Discount Model (DDM)
    fair_value = net_distribution / REQUIRED_YIELD
    
    # Adjust for NAV erosion due to return of capital (ROC)
    adjusted_nav = cony_nav * (1 - NAV_EROSION_FACTOR)
    
    # Final fair value is taken as the lower of the DDM-based value and the adjusted NAV
    final_fair_value = min(fair_value, adjusted_nav)
    
    results = {
        "CONY_NAV": round(cony_nav, 2),
        "COIN_Current_Price": round(coin_price, 2),
        "Expense Ratio": round(expense_ratio, 4),
        "Treasury Yield": round(treasury_yield, 4),
        "Historical Volatility (COIN)": round(volatility, 4),
        "Call Option Premium": round(call_premium, 2),
        "Annual Option Income (Adjusted)": round(annual_option_income, 2),
        "Annual Treasury Income": round(annual_treasury_income, 2),
        "Gross Annual Income": round(gross_income, 2),
        "Annual Expenses": round(annual_expenses, 2),
        "Net Annual Income": round(net_distribution, 2),
        "Fair Value (DDM)": round(fair_value, 2),
        "Adjusted NAV (after erosion)": round(adjusted_nav, 2),
        "Fair Price Premium/(Discount) %": round((final_fair_value - cony_nav) / cony_nav * 100, 2),
        "Final Adjusted Fair Value": round(final_fair_value, 2)
    }
    
    return results

def main():
    try:
        fair_value_results = compute_cony_fair_value()
        print("\nAutomated CONY ETF Fair Valuation Model")
        print("----------------------------------------")
        for key, value in fair_value_results.items():
            print(f"{key}: {value}")
        print("\nNote: This model automates data retrieval for pricing, volatility, expense ratios, and Treasury yields. "
              "It incorporates option risk adjustments and NAV erosion due to return of capital.")
    except Exception as e:
        print(f"An error occurred during valuation: {e}")

if __name__ == "__main__":
    main()
