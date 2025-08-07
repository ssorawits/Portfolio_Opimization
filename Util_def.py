import os
import math
import random 
import pandas as pd
import numpy as np
import datetime as dt
from pandas_datareader import data as pdr
import yfinance as yf
import tensorflow as tf
from scipy.optimize import minimize
# import pandas_ta as ta
from pypfopt import (
    EfficientFrontier,
    risk_models,
    expected_returns,
    objective_functions,
)


def getData(stocks, start, end):
    stock_data = yf.download(stocks, start=start, end=end, group_by='ticker')
    close_data = pd.DataFrame()

    for stock in stocks:
        if stock in stock_data:
            close_data[stock] = stock_data[stock]['Close'].round(2)

    return close_data

# find avg. days per month
def avg_days_per_month(df):
    df = df.copy()
    df['YearMonth'] = df.index.to_period('M')
    monthly_counts = df.groupby('YearMonth').size()
    avg_days = math.ceil(monthly_counts.mean())
    return avg_days

def save_dataframe_to_new_sheet(df: pd.DataFrame, excel_path, sheet_name) -> None:
    # Ensure the directory exists
    os.makedirs(os.path.dirname(excel_path), exist_ok=True)

    try:
        if os.path.exists(excel_path):
            # File exists, append new sheet or overwrite existing one
            with pd.ExcelWriter(excel_path, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
                df.to_excel(writer, sheet_name=sheet_name, index=True)
            print(f"DataFrame saved to sheet '{sheet_name}' in existing file: {excel_path} 📄")
        else:
            # File does not exist, create it and add the sheet
            with pd.ExcelWriter(excel_path, mode='w', engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name=sheet_name, index=True)
            print(f"DataFrame saved to sheet '{sheet_name}' in new file: {excel_path} ✨")
    except Exception as e:
        print(f"An error occurred: {e} ⚠️")

def get_rebalance_dates(data, start_year=2020):
    """Generate quarterly rebalance dates"""
    first_days = (
        data
        .groupby(data.index.to_period("Q"))
        .apply(lambda grp: grp.index.min())
    )
    
    rebalance_dates = [date for date in first_days if date >= dt.datetime(start_year, 1, 1)]
    rebalance_dates_str = [date.strftime('%Y-%m-%d') for date in rebalance_dates]
    
    print(f"Rebalance Dates: {rebalance_dates_str}")
    return rebalance_dates

def get_quarter_end_date(quarter_start, data_index, trading_days_per_quarter=63):
    """Optimized quarter end calculation"""
    try:
        # More efficient position finding
        start_loc = data_index.get_indexer([quarter_start], method='nearest')[0]
        end_loc = min(start_loc + trading_days_per_quarter - 1, len(data_index) - 1)
        return data_index[end_loc]
    except Exception:
        # Fallback method
        nearest_dates = data_index[data_index >= quarter_start]
        if len(nearest_dates) == 0:
            return data_index[-1]
        start_loc = data_index.get_loc(nearest_dates[0])
        end_loc = min(start_loc + trading_days_per_quarter - 1, len(data_index) - 1)
        return data_index[end_loc]

def set_seed(seed_val):
    os.environ['PYTHONHASHSEED'] = str(seed_val)
    tf.random.set_seed(seed_val)
    np.random.seed(seed_val)
    random.seed(seed_val)

def check_portfolio_constraints(weights_df, asset_map, asset_lower, asset_upper):
    # Dictionary to store any violations found
    violations = {}
    # Iterate over each row (date) in the weights DataFrame
    for date, row in weights_df.iterrows():
        # Initialize a dictionary to store the summed weights for each asset class for the current date
        asset_class_weights = {
            'Cash_Equivalent': 0.0,
            'Fixed_Income': 0.0,
            'Equity': 0.0,
            'Alternatives': 0.0
        }
        
        date_violations = []

        # Calculate the total weight for each asset class
        for ticker, weight in row.items():
            asset_class = asset_map.get(ticker)
            if asset_class:
                asset_class_weights[asset_class] += weight
        
        # Check if the calculated weights are within the defined bounds
        for asset_class, total_weight in asset_class_weights.items():
            lower_bound = asset_lower.get(asset_class, -float('inf'))
            upper_bound = asset_upper.get(asset_class, float('inf'))
            
            # Round to avoid floating point inaccuracies
            total_weight = round(total_weight, 2)

            if not (lower_bound <= total_weight <= upper_bound):
                violation_message = (
                    f"Violation in '{asset_class}': "
                    f"Total weight is {total_weight:.2f}, "
                    f"but bounds are [{lower_bound}, {upper_bound}]."
                )
                date_violations.append(violation_message)
        
        if date_violations:
            violations[date.strftime('%Y-%m-%d')] = date_violations
            
    return violations


# --------------------------------------------------------#
# 2. Traditional mean-variance optimization
# def calculate_portfolio_stats(weights, returns):
#     """Calculate portfolio return, volatility, and Sharpe ratio"""
#     portfolio_return = np.sum(returns.mean() * weights) * 252  # Annualized
#     portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
#     sharpe_ratio = (portfolio_return - 0.02) / portfolio_vol if portfolio_vol > 0 else 0
#     return portfolio_return, portfolio_vol, sharpe_ratio

# def objective_function(weights, returns):
#     """Objective function to minimize (negative Sharpe ratio)"""
#     _, _, sharpe = calculate_portfolio_stats(weights, returns)
#     return -sharpe

# def create_asset_class_constraints(asset_symbols, asset_map, asset_lower, asset_upper):
#     """Create asset class weight constraints"""
#     constraints = []
    
#     for asset_class in asset_lower.keys():
#         # Find indices of assets in this class
#         asset_indices = [i for i, symbol in enumerate(asset_symbols) if asset_map[symbol] == asset_class]
        
#         if asset_indices:
#             # Lower bound constraint for asset class
#             constraints.append({
#                 'type': 'ineq',
#                 'fun': lambda x, indices=asset_indices, lower=asset_lower[asset_class]: 
#                     np.sum(x[indices]) - lower
#             })
            
#             # Upper bound constraint for asset class
#             constraints.append({
#                 'type': 'ineq', 
#                 'fun': lambda x, indices=asset_indices, upper=asset_upper[asset_class]: 
#                     upper - np.sum(x[indices])
#             })
    
#     return constraints

# def optimize_portfolio(returns, asset_symbols, asset_map, asset_lower, asset_upper, port_type):
#     """Optimize portfolio weights to maximize Sharpe ratio with asset class constraints"""
#     n_assets = len(returns.columns)
    
#     # Basic constraint: weights sum to 1
#     constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    
#     # Add asset class constraints
#     asset_class_constraints = create_asset_class_constraints(
#         asset_symbols, asset_map, asset_lower, asset_upper
#     )
#     constraints.extend(asset_class_constraints)
    
#     # Bounds for individual assets
#     bounds = tuple(port_type for _ in range(n_assets))
    
#     # Initial guess: equal weights
#     initial_guess = np.array([1/n_assets] * n_assets)
    
#     # Optimize
#     result = minimize(
#         objective_function,
#         initial_guess,
#         args=(returns,),
#         method='SLSQP',
#         bounds=bounds,
#         constraints=constraints,
#         options={'ftol': 1e-9, 'disp': False}
#     )
    
#     return result.x if result.success else initial_guess

# def quarterly_rebalancing(data, asset_map, asset_lower, asset_upper, port_type, 
#                           start_year=2020, trading_days_per_quarter=63, min_train_periods=252):
#     """Perform quarterly mean-variance optimization with asset class constraints"""
#     weights_dict = {}
#     asset_symbols = list(data.columns)
    
#     # for rebal_date in rebalance_dates:
#     #     print(f"Rebalance Date: {rebal_date.strftime('%Y-%m-%d')}")
        
#     #     # Get lookback period (1 year of data before rebalance date)
#     #     lookback_start = rebal_date - pd.DateOffset(days=365)
#     #     lookback_data = data[(data.index >= lookback_start) & (data.index < rebal_date)]
#     #     print(f"Lookback period: {lookback_start.strftime('%Y-%m-%d')} to {rebal_date.strftime('%Y-%m-%d')}")
#     #     print(f"Lookback data shape: {lookback_data.shape}")

#     rebalance_dates = get_rebalance_dates(data, start_year)
#     # Store original datetime index
#     original_index = data.index.copy()
#     # Reset index to integer for processing
#     df_reset = data.copy()
#     df_reset.index = range(len(df_reset))

#     for i, rebalance_date in enumerate(rebalance_dates):
#         print(f"\n--- Rebalancing {i+1}/{len(rebalance_dates)} ---")
#         print(f"Rebalance Date: {rebalance_date.strftime('%Y-%m-%d')}")
        
#         # Find corresponding position in reset dataframe
#         try:
#             train_end_pos = original_index.get_loc(rebalance_date) - 1
#         except KeyError:
#             # Find nearest date if exact date doesn't exist
#             nearest_dates = original_index[original_index >= rebalance_date]
#             if len(nearest_dates) == 0:
#                 print(f"No data available for {rebalance_date}, skipping...")
#                 continue
#             train_end_pos = original_index.get_loc(nearest_dates[0])
        
#         # Define training period - 1 day before rebalance date
#         train_start_pos = max(0, train_end_pos - min_train_periods)
        
#         print(f"Training period: {original_index[train_start_pos].strftime('%Y-%m-%d')} to {original_index[train_end_pos].strftime('%Y-%m-%d')}")
#         print(f"Training days: {train_end_pos - train_start_pos + 1}")
        
#         # Get quarter end date
#         quarter_end = get_quarter_end_date(rebalance_date, original_index, trading_days_per_quarter)
        
#         try:
#             test_end_pos = original_index.get_loc(quarter_end)
#             print(f"Test period: {rebalance_date.strftime('%Y-%m-%d')} to {quarter_end.strftime('%Y-%m-%d')}")
#             print(f"Test days: {test_end_pos - train_end_pos}")
#         except:
#             print(f"Cannot find end date for quarter, skipping...")
#             continue
        
#         # Skip if we don't have enough future data
#         if test_end_pos >= len(df_reset):
#             print(f"Not enough future data, stopping at rebalance {i+1}")
#             break
            
#         # Extract training data
#         train_data = df_reset.iloc[train_start_pos:train_end_pos+1].copy()
#         # train_features = features_reset.iloc[train_start_pos:train_end_pos+1].copy()
        
#         # Fill missing values
#         train_data = train_data.ffill().bfill()
#         # train_features = train_features.fillna(0)


#         # if len(lookback_data) < 50:  # Need sufficient data
#         #     print(f"Insufficient data for {rebal_date}")
#         #     continue
        
#         lookback_data = train_data.copy()
#         if len(lookback_data) < 50:  # Need sufficient data
#             print(f"Insufficient data for {rebalance_date}")
#             continue
            
#         # Calculate returns for lookback period
#         returns = lookback_data.pct_change().dropna()
        
#         # Optimize portfolio with asset class constraints
#         optimal_weights = optimize_portfolio(
#             returns, asset_symbols, asset_map, asset_lower, asset_upper, port_type
#         )
        
#         # Store weights for this rebalance date
#         weights_dict[rebalance_date] = optimal_weights
        
#         # Calculate and print stats
#         port_return, port_vol, sharpe = calculate_portfolio_stats(optimal_weights, returns)
#         print(f"Sharpe Ratio: {sharpe:.4f}")
        
#         # Print asset class allocations
#         asset_class_weights = {}
#         for asset_class in asset_lower.keys():
#             class_weight = sum(optimal_weights[i] for i, symbol in enumerate(asset_symbols) 
#                              if asset_map[symbol] == asset_class)
#             asset_class_weights[asset_class] = class_weight
#             print(f"{asset_class}: {class_weight:.4f}")
    
#     # Create DataFrame with rebalance_dates as index and ETF symbols as columns
#     optimal_weights_df = pd.DataFrame.from_dict(
#         weights_dict, 
#         orient='index', 
#         columns=data.columns
#     )
    
#     return optimal_weights_df

def mvo_quarterly_rebalancing(data, asset_map, asset_lower, asset_upper, port_type, 
                          start_year=2020, trading_days_per_quarter=63, min_train_periods=252, risk_free_rate=0.02):
    weights_dict = {}
    asset_symbols = list(data.columns)
    
    rebalance_dates = get_rebalance_dates(data, start_year)
    # Store original datetime index
    original_index = data.index.copy()
    # Reset index to integer for processing
    df_reset = data.copy()
    df_reset.index = range(len(df_reset))

    for i, rebalance_date in enumerate(rebalance_dates):
        print(f"\n--- Rebalancing {i+1}/{len(rebalance_dates)} ---")
        print(f"Rebalance Date: {rebalance_date.strftime('%Y-%m-%d')}")
        
        # Find corresponding position in reset dataframe
        try:
            train_end_pos = original_index.get_loc(rebalance_date) - 1
        except KeyError:
            # Find nearest date if exact date doesn't exist
            nearest_dates = original_index[original_index >= rebalance_date]
            if len(nearest_dates) == 0:
                print(f"No data available for {rebalance_date}, skipping...")
                continue
            train_end_pos = original_index.get_loc(nearest_dates[0])
        
        # Define training period - 1 day before rebalance date
        train_start_pos = max(0, train_end_pos - min_train_periods)
        
        print(f"Training period: {original_index[train_start_pos].strftime('%Y-%m-%d')} to {original_index[train_end_pos].strftime('%Y-%m-%d')}")
        print(f"Training days: {train_end_pos - train_start_pos + 1}")
        
        # Get quarter end date
        quarter_end = get_quarter_end_date(rebalance_date, original_index, trading_days_per_quarter)
        
        # Extract training data
        train_data = df_reset.iloc[train_start_pos:train_end_pos+1].copy()
        
        train_data = train_data.ffill().bfill()
        
        lookback_data = train_data.copy()
        if len(lookback_data) < 50:  # Need sufficient data
            print(f"❌ Insufficient data for {rebalance_date}")
            continue
            
        try:
            mu = expected_returns.mean_historical_return(lookback_data)
            S = risk_models.sample_cov(lookback_data)
        except Exception as e:
            print(f"❌ Could not calculate returns/covariance: {e}")
            continue

        # Set up the optimizer
        ef_agg = EfficientFrontier(mu, S, weight_bounds=port_type)
        ef_agg.add_sector_constraints(asset_map, asset_lower, asset_upper)
        ef_agg.add_constraint(lambda x: x <= 0.3)

        # Find the portfolio that maximizes the Sharpe ratio
        try:
            weights_agg = ef_agg.max_sharpe(risk_free_rate=risk_free_rate)
            cleaned_weights_agg = ef_agg.clean_weights()
        except Exception as e:
            print(f"❌ Optimization failed for {rebalance_date.strftime('%Y-%m-%d')}: {e}")
            continue
        
        # Store weights for this rebalance date
        weights_dict[rebalance_date] = cleaned_weights_agg
        
        print("\nOptimal Weights:")
        # Print asset class allocations
        asset_class_weights = {}
        # Iterate over the asset classes to calculate their total weight
        for asset_class in sorted(asset_lower.keys()):
            class_weight = sum(cleaned_weights_agg[symbol] for symbol in asset_symbols
                               if asset_map.get(symbol) == asset_class)
            asset_class_weights[asset_class] = class_weight
            print(f"{asset_class}: {class_weight:.4f}")
    
    # Create DataFrame with rebalance_dates as index and ETF symbols as columns
    optimal_weights_df = pd.DataFrame.from_dict(
        weights_dict, 
        orient='index'
    )
    # Ensure all original assets are columns, filling missing with 0
    optimal_weights_df = optimal_weights_df.reindex(columns=data.columns, fill_value=0)
    
    return optimal_weights_df

def mvo_quarterly_rebalancing_2(data, asset_map, asset_lower, asset_upper, port_type, 
                          start_year=2020, trading_days_per_quarter=63, min_train_periods=252, risk_free_rate=0.02):
    weights_dict = {}
    asset_symbols = list(data.columns)
    ETF_list = list(data.columns)
    rebalance_dates = get_rebalance_dates(data, start_year)
    # Store original datetime index
    original_index = data.index.copy()
    # Reset index to integer for processing
    df_reset = data.copy()
    df_reset.index = range(len(df_reset))

    for i, rebalance_date in enumerate(rebalance_dates):
        print(f"\n--- Rebalancing {i+1}/{len(rebalance_dates)} ---")
        print(f"Rebalance Date: {rebalance_date.strftime('%Y-%m-%d')}")
        
        # Find corresponding position in reset dataframe
        try:
            train_end_pos = original_index.get_loc(rebalance_date) - 1
        except KeyError:
            # Find nearest date if exact date doesn't exist
            nearest_dates = original_index[original_index >= rebalance_date]
            if len(nearest_dates) == 0:
                print(f"No data available for {rebalance_date}, skipping...")
                continue
            train_end_pos = original_index.get_loc(nearest_dates[0])
        
        # Define training period - 1 day before rebalance date
        train_start_pos = max(0, train_end_pos - min_train_periods)
        
        print(f"Training period: {original_index[train_start_pos].strftime('%Y-%m-%d')} to {original_index[train_end_pos].strftime('%Y-%m-%d')}")
        print(f"Training days: {train_end_pos - train_start_pos + 1}")
        
        # Get quarter end date
        quarter_end = get_quarter_end_date(rebalance_date, original_index, trading_days_per_quarter)
        
        # Extract training data
        train_data = df_reset.iloc[train_start_pos:train_end_pos+1].copy()
        
        train_data = train_data.ffill().bfill()
        
        lookback_data = train_data.copy()
        if len(lookback_data) < 50:  # Need sufficient data
            print(f"❌ Insufficient data for {rebalance_date}")
            continue
            
        mu = expected_returns.mean_historical_return(lookback_data)
        S = risk_models.CovarianceShrinkage(lookback_data).ledoit_wolf()

        # Set up the optimizer
        candidate_portfolios = []
        gamma_start = 0.0
        gamma_end = 1.0
        gamma_step = 0.005
        gamma_range = [round(g, 4) for g in np.arange(gamma_start, gamma_end, gamma_step)]

        for gamma_val in gamma_range:
            ef = EfficientFrontier(mu, S, weight_bounds=(0, 1))
            # --- ใส่เงื่อนไขทั้งหมด ---
            ef.add_objective(objective_functions.L2_reg, gamma=gamma_val)

            weight_upper_bounds = {ticker: 0.30 for ticker in ETF_list}
            if 'SHV' in weight_upper_bounds:
                weight_upper_bounds['SHV'] = 0.40
            custom_bounds = [(0, weight_upper_bounds[ticker]) for ticker in ef.tickers]
            ef.weight_bounds = custom_bounds

            asset_lower_aggressive = {'Cash_Equivalent': 0.0, 'Fixed_Income': 0.0, 'Equity': 0.55, 'Alternatives': 0.0}
            asset_upper_aggressive = {'Cash_Equivalent': 0.4, 'Fixed_Income': 0.3, 'Equity': 0.9, 'Alternatives': 0.3}
            ef.add_sector_constraints(asset_map, asset_lower_aggressive, asset_upper_aggressive)

            try:
                weights = ef.max_sharpe()
                cleaned_weights = ef.clean_weights()
                num_assets = len([w for w in cleaned_weights.values() if w > 0])

                if 10 <= num_assets <= 15:
                    performance = ef.portfolio_performance()
                    sharpe = performance[2]
                    
                    candidate_portfolios.append({
                        "gamma": gamma_val,
                        "num_assets": num_assets,
                        "sharpe_ratio": sharpe,
                        "weights": cleaned_weights
                    })
                        # print(f"Gamma {gamma_val:.3f} -> {num_assets} assets. Sharpe: {sharpe:.2f}. (Found Candidate!)")
            except ValueError:
                continue

        print("\n" + "="*60)
        print("Search complete.")

        if not candidate_portfolios:
            print("⚠️ No portfolio found matching the 10-15 asset constraint.")
            print("Please try adjusting the gamma search range or constraints.")
        else:
            print(f"Found {len(candidate_portfolios)} candidate portfolios.")
            for n_assets in range(10, 16):
                filtered = [p for p in candidate_portfolios if p['num_assets'] == n_assets]
                print(f"Checking portfolios with {n_assets} assets... Found {len(filtered)} candidates.")
                if filtered:
                    best_portfolio = max(filtered, key=lambda p: p['sharpe_ratio'])
                    break
            else:
                best_portfolio = max(candidate_portfolios, key=lambda p: p['sharpe_ratio'])

            # final_weights = pd.Series(best_portfolio['weights'])
        
        # Store weights for this rebalance date
        weights_dict[rebalance_date] = best_portfolio['weights']
        
    
    # Create DataFrame with rebalance_dates as index and ETF symbols as columns
    optimal_weights_df = pd.DataFrame.from_dict(
        weights_dict, 
        orient='index'
    )
    # Ensure all original assets are columns, filling missing with 0
    optimal_weights_df = optimal_weights_df.reindex(columns=data.columns, fill_value=0)
    
    return optimal_weights_df


# --------------------------------------------------------#
# 3. Equal weights
def equal_weight_portfolio(data, rebalance_dates):
   """Calculate equal weight portfolio based on rebalance dates"""
   weights_dict = {}
   n_assets = len(data.columns)
   equal_weight = 1.0 / n_assets
   
   for rebal_date in rebalance_dates:
       # Equal weights for all assets
       weights_dict[rebal_date] = [equal_weight] * n_assets
   
   # Create DataFrame with rebalance_dates as index and ETF symbols as columns
   equal_weights_df = pd.DataFrame.from_dict(
       weights_dict, 
       orient='index', 
       columns=data.columns
   )
   
   return equal_weights_df

# --------------------------------------------------------#
# All Compare portfolios
def calculate_portfolio_returns(data, weights_df, rebalance_dates, start_date):
    weights_df_2 = weights_df.copy()
    """Calculate portfolio returns with rebalancing"""
    portfolio_returns = []
    
    # Filter data from start date
    data_filtered = data[data.index >= start_date]
    
    for i in range(len(data_filtered)):
        current_date = data_filtered.index[i]
        
        # Find the most recent rebalance date
        applicable_rebalance = None
        for rebal_date in rebalance_dates:
            if rebal_date <= current_date:
                applicable_rebalance = rebal_date
            else:
                break
        
        if applicable_rebalance is None:
            continue
            
        # Get weights for the applicable rebalance date
        rebal_date_str = applicable_rebalance.strftime('%Y-%m-%d')
        if rebal_date_str in weights_df_2.index:
            weights = weights_df_2.loc[rebal_date_str]
            
            # Calculate daily returns
            if i > 0:
                prev_prices = data_filtered.iloc[i-1]
                curr_prices = data_filtered.iloc[i]
                daily_returns = (curr_prices - prev_prices) / prev_prices
                
                # Calculate portfolio return
                portfolio_return = (weights * daily_returns).sum()
                portfolio_returns.append(portfolio_return)
            else:
                portfolio_returns.append(0)
    
    return pd.Series(portfolio_returns, index=data_filtered.index[1:] if len(portfolio_returns) == len(data_filtered)-1 else data_filtered.index)

def calculate_max_drawdown(returns):
    """Calculate maximum drawdown and duration"""
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdown = (cumulative - rolling_max) / rolling_max
    
    max_dd = drawdown.min()
    
    # Calculate drawdown duration
    is_drawdown = drawdown < 0
    drawdown_periods = []
    start = None
    
    for i, in_dd in enumerate(is_drawdown):
        if in_dd and start is None:
            start = i
        elif not in_dd and start is not None:
            drawdown_periods.append(i - start)
            start = None
    
    if start is not None:
        drawdown_periods.append(len(is_drawdown) - start)
    
    max_dd_duration = max(drawdown_periods) if drawdown_periods else 0
    
    return max_dd * 100, max_dd_duration

def calculate_performance_metrics(returns, benchmark_returns, risk_free_rate=0.02):
    """Calculate comprehensive performance metrics"""
    # Basic metrics
    total_return = (1 + returns).prod() - 1
    annualized_return = (1 + returns).prod() ** (252 / len(returns)) - 1
    volatility = returns.std() * np.sqrt(252)
    
    # Max drawdown
    max_dd, max_dd_duration = calculate_max_drawdown(returns)
    
    # Risk-adjusted metrics
    excess_returns = returns - risk_free_rate/252
    sharpe_ratio = excess_returns.mean() / returns.std() * np.sqrt(252)
    
    # Sortino ratio (downside deviation)
    downside_returns = returns[returns < 0]
    sortino_ratio = excess_returns.mean() / downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else np.nan
    
    # Beta and Alpha (relative to benchmark)
    if len(benchmark_returns) == len(returns):
        covariance = np.cov(returns, benchmark_returns)[0][1]
        benchmark_variance = np.var(benchmark_returns)
        beta = covariance / benchmark_variance if benchmark_variance != 0 else np.nan
        
        benchmark_excess = benchmark_returns - risk_free_rate/252
        alpha = (returns - risk_free_rate/252).mean() - beta * benchmark_excess.mean()
        alpha_annualized = alpha * 252
        
        # R-squared
        correlation = np.corrcoef(returns, benchmark_returns)[0][1]
        r_squared = correlation ** 2
        
        # Treynor ratio
        treynor_ratio = excess_returns.mean() * 252 / beta if beta != 0 else np.nan
        
        # Tracking error
        tracking_error = (returns - benchmark_returns).std() * np.sqrt(252)
        
        # Information ratio
        active_return = returns.mean() - benchmark_returns.mean()
        information_ratio = active_return * 252 / tracking_error if tracking_error != 0 else np.nan
    else:
        beta = alpha_annualized = r_squared = treynor_ratio = tracking_error = information_ratio = np.nan
    
    # Value at Risk (95% confidence)
    var_95 = np.percentile(returns, 5) * 100
    
    return {
        'Total Return (%)': total_return * 100,
        'Annualized Return (%)': annualized_return * 100,
        'Volatility (%)': volatility * 100,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown (%)': max_dd,
        'Max Drawdown Duration (days)': max_dd_duration,
        # 'Sharpe Ratio': sharpe_ratio,
        'Sortino Ratio': sortino_ratio,
        'Treynor Ratio': treynor_ratio,
        'Jensen\'s Alpha (%)': alpha_annualized * 100 if not np.isnan(alpha_annualized) else np.nan,
        'Beta': beta,
        'R-squared': r_squared,
        'Tracking Error (%)': tracking_error * 100 if not np.isnan(tracking_error) else np.nan,
        'Information Ratio': information_ratio,
        'Value at Risk (VaR) (%)': var_95
    }

# --------------------------------------------------------#
# Quarterly Compare portfolios
def get_quarterly_periods(rebalance_dates, data):
    """Define quarterly periods based on rebalance dates"""
    periods = []
    
    for i in range(len(rebalance_dates)):
        start_date = rebalance_dates[i]
        
        # End date is the day before next rebalance (or end of data)
        if i < len(rebalance_dates) - 1:
            end_date = rebalance_dates[i + 1] - pd.Timedelta(days=1)
        else:
            end_date = data.index[-1]
        
        # Create quarter label
        quarter_label = f"Q{((start_date.month - 1) // 3) + 1} {start_date.year}"
        
        periods.append({
            'quarter': quarter_label,
            'start_date': start_date,
            'end_date': end_date,
            'rebalance_date': start_date
        })
    
    return periods

def calculate_quarterly_portfolio_returns(data, weights_df, period_info):
    weights_df_2 = weights_df.copy()
    """Calculate portfolio returns for a specific quarter"""
    start_date = period_info['start_date']
    end_date = period_info['end_date']
    rebalance_date_str = period_info['rebalance_date'].strftime('%Y-%m-%d')
    
    # Get data for this period
    period_data = data[(data.index >= start_date) & (data.index <= end_date)]
    
    if len(period_data) < 2:
        return pd.Series(dtype=float)
    
    # Get weights for this rebalance date
    if rebalance_date_str not in weights_df_2.index:
        return pd.Series(dtype=float)
    
    weights = weights_df_2.loc[rebalance_date_str]
    
    # Calculate daily returns
    daily_returns = period_data.pct_change().dropna()
    
    # Calculate portfolio returns
    portfolio_returns = (daily_returns * weights).sum(axis=1)
    
    return portfolio_returns

def calculate_quarterly_metrics(returns):
    """Calculate performance metrics for quarterly returns"""
    if len(returns) == 0:
        return {
            'Total Return (%)': np.nan,
            'Annualized Return (%)': np.nan,
            'Volatility (%)': np.nan,
            'Max Drawdown (%)': np.nan,
            'Max Drawdown Duration (days)': np.nan,
            'Sharpe Ratio': np.nan,
            'Sortino Ratio': np.nan,
        }
    
    # Total return for the quarter
    total_return = (1 + returns).prod() - 1
    
    # Annualized return
    annualized_return = (1 + returns).prod() ** (252 / len(returns)) - 1

    # Volatility (annualized)
    volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else np.nan
    
    # Max drawdown
    max_dd, max_dd_duration = calculate_max_drawdown(returns)

    # Sharpe ratio (assuming 2% risk-free rate)
    risk_free_rate = 0.02
    excess_returns = returns - risk_free_rate/252
    sharpe_ratio = excess_returns.mean() / returns.std() * np.sqrt(252) if returns.std() != 0 else np.nan

    # Sortino ratio (downside deviation)
    downside_returns = returns[returns < 0]
    sortino_ratio = excess_returns.mean() / downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else np.nan
    
    # Max drawdown for the quarter
    if len(returns) > 0:
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_dd = drawdown.min() * 100
    else:
        max_dd = np.nan
    
    return {
        'Total Return (%)': total_return * 100,
        'Annualized Return (%)': annualized_return * 100,
        'Volatility (%)': volatility * 100,
        'Max Drawdown (%)': max_dd,
        'Max Drawdown Duration (days)': max_dd_duration,
        'Sharpe Ratio': sharpe_ratio,
        'Sortino Ratio': sortino_ratio,
    }

# --------------------------------------------------------#
# Yearly Comparison

def get_yearly_periods(rebalance_dates, data):
    """Define yearly periods based on data availability"""
    periods = []
    
    # Get unique years from rebalance dates and data
    start_year = rebalance_dates[0].year
    end_year = data.index[-1].year
    
    for year in range(start_year, end_year + 1):
        # Find start date for this year
        if year == start_year:
            start_date = rebalance_dates[0]  # First rebalance date
        else:
            start_date = pd.Timestamp(f'{year}-01-01')
            # Find the first available date in data for this year
            year_data = data[data.index.year == year]
            if len(year_data) > 0:
                start_date = year_data.index[0]
            else:
                continue
        
        # Find end date for this year
        if year == end_year:
            end_date = data.index[-1]  # Last available date
        else:
            end_date = pd.Timestamp(f'{year}-12-31')
            # Find the last available date in data for this year
            year_data = data[data.index.year == year]
            if len(year_data) > 0:
                end_date = year_data.index[-1]
            else:
                continue
        
        periods.append({
            'year': year,
            'start_date': start_date,
            'end_date': end_date
        })
    
    return periods

def get_applicable_weights(weights_df, target_date, rebalance_dates):
    weights_df_2 = weights_df.copy()
    """Find the most recent rebalance weights for a given date"""
    applicable_rebalance = None
    
    for rebal_date in rebalance_dates:
        if rebal_date <= target_date:
            applicable_rebalance = rebal_date
        else:
            break
    
    if applicable_rebalance is None:
        return None
    
    rebal_date_str = applicable_rebalance.strftime('%Y-%m-%d')
    if rebal_date_str in weights_df_2.index:
        return weights_df_2.loc[rebal_date_str]
    
    return None

def calculate_yearly_portfolio_returns(data, weights_df, period_info, rebalance_dates):
    weights_df_2 = weights_df.copy()
    """Calculate portfolio returns for a specific year with rebalancing"""
    start_date = period_info['start_date']
    end_date = period_info['end_date']
    
    # Get data for this period
    period_data = data[(data.index >= start_date) & (data.index <= end_date)]
    
    if len(period_data) < 2:
        return pd.Series(dtype=float)
    
    portfolio_returns = []
    
    for i in range(1, len(period_data)):
        current_date = period_data.index[i]
        
        # Get applicable weights for this date
        weights = get_applicable_weights(weights_df_2, current_date, rebalance_dates)
        
        if weights is None:
            continue
        
        # Calculate daily return
        prev_prices = period_data.iloc[i-1]
        curr_prices = period_data.iloc[i]
        daily_returns = (curr_prices - prev_prices) / prev_prices
        
        # Calculate portfolio return
        portfolio_return = (weights * daily_returns).sum()
        portfolio_returns.append(portfolio_return)
    
    return pd.Series(portfolio_returns, index=period_data.index[1:len(portfolio_returns)+1])

def calculate_yearly_metrics(returns):
    """Calculate comprehensive performance metrics for yearly returns"""
    if len(returns) == 0:
        return {
            'Total Return (%)': np.nan,
            'Annualized Return (%)': np.nan,
            'Volatility (%)': np.nan,
            'Sharpe Ratio': np.nan,
            'Sortino Ratio': np.nan,
            'Max Drawdown (%)': np.nan,
            'Max DD Duration (days)': np.nan,
            'VaR 95% (%)': np.nan,
            'Calmar Ratio': np.nan,
            'Trading Days': len(returns)
        }
    
    # Total return for the year
    total_return = (1 + returns).prod() - 1
    
    # Annualized return (if less than a year, adjust accordingly)
    trading_days = len(returns)
    annualized_return = (1 + total_return) ** (252 / trading_days) - 1
    
    # Volatility (annualized)
    volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else np.nan
    
    # Risk-adjusted metrics
    risk_free_rate = 0.02
    excess_returns = returns - risk_free_rate/252
    sharpe_ratio = excess_returns.mean() / returns.std() * np.sqrt(252) if returns.std() != 0 else np.nan
    
    # Sortino ratio (downside deviation)
    downside_returns = returns[returns < 0]
    sortino_ratio = excess_returns.mean() / downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else np.nan
    
    # Max drawdown and duration
    if len(returns) > 0:
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_dd = drawdown.min() * 100
        
        # Drawdown duration
        is_drawdown = drawdown < -0.001  # 0.1% threshold
        drawdown_periods = []
        start = None
        
        for i, in_dd in enumerate(is_drawdown):
            if in_dd and start is None:
                start = i
            elif not in_dd and start is not None:
                drawdown_periods.append(i - start)
                start = None
        
        if start is not None:
            drawdown_periods.append(len(is_drawdown) - start)
        
        max_dd_duration = max(drawdown_periods) if drawdown_periods else 0
    else:
        max_dd = np.nan
        max_dd_duration = np.nan
    
    # Value at Risk (95% confidence)
    var_95 = np.percentile(returns, 5) * 100 if len(returns) > 0 else np.nan
    
    # Calmar Ratio (annualized return / max drawdown)
    calmar_ratio = annualized_return * 100 / abs(max_dd) if max_dd != 0 and not np.isnan(max_dd) else np.nan
    
    return {
        'Total Return (%)': total_return * 100,
        'Annualized Return (%)': annualized_return * 100,
        'Volatility (%)': volatility * 100,
        'Sharpe Ratio': sharpe_ratio,
        'Sortino Ratio': sortino_ratio,
        'Max Drawdown (%)': max_dd,
        'Max DD Duration (days)': max_dd_duration,
        'VaR 95% (%)': var_95,
        'Calmar Ratio': calmar_ratio,
        'Trading Days': trading_days
    }



def create_direct_forecast_dataset(features, prices, sequence_length, forecast_horizon):
    """
    สร้างชุดข้อมูลสำหรับ Direct Multi-step Forecast

    Args:
        features (pd.DataFrame): DataFrame ของ features สำหรับใช้เป็น input (X)
        prices (pd.DataFrame): DataFrame ของราคาสินทรัพย์ สำหรับสร้าง target (y)
        sequence_length (int): จำนวนวันย้อนหลังที่จะใช้เป็น Input (X)
        forecast_horizon (int): จำนวนวันในอนาคตที่ต้องการพยากรณ์ (y)

    Returns:
        tuple: (X, y) ที่เป็น NumPy arrays
            - X: (n_samples, sequence_length, n_features)
            - y: (n_samples, forecast_horizon, n_assets) -> เป็น daily returns
    """
    x_data, y_data = [], []
    feature_values = features.values
    price_values = prices.values
    
    # วนลูปเพื่อสร้าง "หน้าต่าง" ของข้อมูล
    for i in range(len(feature_values) - sequence_length - forecast_horizon + 1):
        # ส่วนของ Input (X) คือ features ย้อนหลัง
        x_slice = feature_values[i : i + sequence_length]
        x_data.append(x_slice)
        
        # ส่วนของ Target (y) คือ returns ในอนาคต
        # 1. ดึงราคาสินทรัพย์ในช่วงเวลาพยากรณ์
        price_slice_future = price_values[i + sequence_length : i + sequence_length + forecast_horizon]
        
        # 2. คำนวณ daily returns จากราคา
        # เพิ่มข้อมูลวันสุดท้ายของ input window เข้ามาเพื่อคำนวณ return ของวันแรกใน horizon
        price_slice_for_calc = np.vstack([price_values[i + sequence_length - 1], price_slice_future])
        
        # 3. คำนวณ % change
        returns_slice = (price_slice_for_calc[1:] - price_slice_for_calc[:-1]) / price_slice_for_calc[:-1]
        np.nan_to_num(returns_slice, copy=False, nan=0.0, posinf=0.0, neginf=0.0) # จัดการค่า NaN/inf
        
        y_data.append(returns_slice)
        
    return np.array(x_data), np.array(y_data)

# def add_technical_features(prices_df: pd.DataFrame) -> pd.DataFrame:
#     """
#     รับ DataFrame ของราคา และคืนค่า DataFrame ที่มี Technical Features เพิ่มเข้ามา
    
#     Args:
#         prices_df (pd.DataFrame): DataFrame ที่มีราคาสินทรัพย์ (แต่ละคอลัมน์คือสินทรัพย์)

#     Returns:
#         pd.DataFrame: DataFrame ของ features ที่พร้อมใช้งาน
#     """
#     features = pd.DataFrame(index=prices_df.index)
    
#     # # สร้าง 'market' proxy เป็นค่าเฉลี่ยของสินทรัพย์ทั้งหมดเพื่อใช้คำนวณ Beta
#     # market_returns = prices_df.pct_change().mean(axis=1)

#     # วนลูปเพื่อสร้าง feature สำหรับแต่ละสินทรัพย์
#     for col in prices_df.columns:
#         price = prices_df[col]
#         returns = price.pct_change()

#         # === กลุ่มที่ 1: ชุดพื้นฐานที่ต้องมี ===
#         # EMA
#         features[f'{col}_EMA_12'] = ta.ema(price, length=12)
#         features[f'{col}_EMA_26'] = ta.ema(price, length=26)
#         features[f'{col}_EMA_50'] = ta.ema(price, length=50)
#         features[f'{col}_EMA_200'] = ta.ema(price, length=200)
        
#         # RSI
#         features[f'{col}_RSI_14'] = ta.rsi(price, length=14)
        
#         # MACD
#         macd = ta.macd(price, fast=12, slow=26, signal=9)
#         features[f'{col}_MACD'] = macd['MACD_12_26_9']
#         features[f'{col}_MACD_signal'] = macd['MACDs_12_26_9']
#         features[f'{col}_MACD_hist'] = macd['MACDh_12_26_9']
        
#         # Rolling Volatility
#         features[f'{col}_VOL_21'] = returns.rolling(window=21).std()
        
#         # ATR
#         # ต้องใช้ high, low, close; เราจะประมาณค่าจากราคา close
#         features[f'{col}_ATR_14'] = ta.atr(high=price, low=price, close=price, length=14)

#         # === กลุ่มที่ 2: ชุดขั้นสูงเพื่อเพิ่มความลึกซึ้ง ===
#         # Bollinger Bands %B
#         bbands = ta.bbands(price, length=20, std=2)
#         features[f'{col}_BB_PERCENT'] = bbands['BBP_20_2.0']

#         # Slope of EMA 50
#         features[f'{col}_EMA_50_SLOPE'] = features[f'{col}_EMA_50'].diff()

#         # # Rolling Beta
#         # rolling_cov = returns.rolling(window=63).cov(market_returns)
#         # rolling_var = market_returns.rolling(window=63).var()
#         # features[f'{col}_BETA_63'] = rolling_cov / rolling_var
        
#         # Rolling Max Drawdown
#         rolling_max = price.rolling(window=63, min_periods=1).max()
#         drawdown = (price - rolling_max) / rolling_max
#         features[f'{col}_MDD_63'] = drawdown

#         # === กลุ่มที่ 3: ชุดสำหรับผู้เชี่ยวชาญ ===
#         # Rolling Skewness & Kurtosis
#         features[f'{col}_SKEW_63'] = returns.rolling(window=63).skew()
#         features[f'{col}_KURT_63'] = returns.rolling(window=63).kurt()
        
#         # Volatility of Volatility
#         features[f'{col}_VOV_21'] = features[f'{col}_VOL_21'].rolling(window=21).std()

#         # === เพิ่ม Return พื้นฐานเข้าไปด้วย ===
#         features[f'{col}_RETURN'] = returns

#     return features


def create_mpt_features(price_data, lookback_window=63):
    """
    สร้าง Feature 2 ชุดตามหลัก MPT:
    1. Returns Features (สำหรับ Query): สะท้อนศักยภาพผลตอบแทน
    2. Risk Features (สำหรับ Key/Value): สะท้อนความเสี่ยงและความสัมพันธ์
    """
    print("Creating MPT-based features...")
    # --- ชุดที่ 1: Returns Features ---
    returns_features = pd.DataFrame(index=price_data.index)
    returns = price_data.pct_change()

    # Feature: Momentum/Returns ระยะสั้น, กลาง, ยาว
    for n in [5, 21, 63]:
        returns_features = returns_features.join(returns.rolling(window=n).mean().add_suffix(f'_ret_{n}d'))

    # --- ชุดที่ 2: Risk & Correlation Features ---
    risk_features = pd.DataFrame(index=price_data.index)

    # Feature: Volatility ของแต่ละสินทรัพย์
    for n in [21, 63]:
        risk_features = risk_features.join(returns.rolling(window=n).std().add_suffix(f'_vol_{n}d'))
        
    # Feature: Correlation เฉลี่ย (Proxy สำหรับ Diversification)
    # คำนวณ Correlation Matrix แบบ Rolling
    corr_matrix_rolling = returns.rolling(window=lookback_window).corr()
    
    avg_corr_data = {}
    for asset in price_data.columns:
        # ดึง correlation ของ asset นั้นๆ กับตัวอื่น แล้วหาค่าเฉลี่ย
        # ไม่รวม correlation กับตัวเอง (ซึ่งเป็น 1)
        avg_corr = corr_matrix_rolling.unstack(1)[asset].drop(columns=asset).mean(axis=1)
        avg_corr_data[f'{asset}_avg_corr'] = avg_corr

    avg_corr_df = pd.DataFrame(avg_corr_data).reset_index().rename(columns={'level_0': 'Date'}).set_index('Date')
    risk_features = risk_features.join(avg_corr_df)
    
    # เติมค่าว่าง
    returns_features.fillna(0, inplace=True)
    risk_features.fillna(0, inplace=True)

    print("MPT features created.")
    return returns_features, risk_features

def create_mpt_enriched_features(price_data, lookback_window=63):
    """
    สร้าง Feature Vector ชุดเดียวที่สมบูรณ์ตามหลัก MPT
    โดยแต่ละคอลัมน์จะแทน feature ของสินทรัพย์แต่ละตัว
    """
    print("Creating MPT-Enriched features...")
    enriched_features_list = []
    
    returns = price_data.pct_change()

    # --- ส่วนที่ 1: มิติผลตอบแทน (Return Component) ---
    print("  - Calculating Return/Momentum features...")
    for n in [5, 21, 63]: # Momentum ระยะสั้น, กลาง, ยาว
        feature_df = returns.rolling(window=n, min_periods=1).mean().add_suffix(f'_ret_{n}d')
        enriched_features_list.append(feature_df)

    # --- ส่วนที่ 2: มิติความเสี่ยงเฉพาะตัว (Individual Risk Component) ---
    print("  - Calculating Risk/Volatility features...")
    for n in [21, 63]: # Volatility ระยะสั้น, กลาง
        feature_df = returns.rolling(window=n, min_periods=1).std().add_suffix(f'_vol_{n}d')
        enriched_features_list.append(feature_df)

    # --- ส่วนที่ 3: มิติความสัมพันธ์ (Correlation/Diversification Component) ---
    print("  - Calculating Correlation features...")
    # การคำนวณ Correlation Matrix แบบ Rolling ใช้ทรัพยากรสูง อาจจะช้าหน่อย
    corr_matrix_rolling = returns.rolling(window=lookback_window, min_periods=lookback_window//2).corr(pairwise=True)
    
    avg_corr_data = {}
    for asset in price_data.columns:
        # ดึง correlation ของ asset นั้นๆ กับตัวอื่น แล้วหาค่าเฉลี่ย
        # ไม่รวม correlation กับตัวเอง (ซึ่งเป็น 1)
        # เราใช้ .xs() เพื่อเลือก cross-section ของ asset นั้นๆ จาก multi-index
        # จากนั้น drop คอลัมน์ตัวเอง แล้วหาค่าเฉลี่ย
        if asset in corr_matrix_rolling.index.get_level_values(1):
             avg_corr = corr_matrix_rolling.xs(asset, level=1).drop(columns=asset, errors='ignore').mean(axis=1)
             avg_corr_data[f'{asset}_avg_corr'] = avg_corr

    avg_corr_df = pd.DataFrame(avg_corr_data)
    # เนื่องจาก avg_corr_df มี MultiIndex (Date, Asset) เราต้อง unstack มัน
    if not avg_corr_df.empty:
       if isinstance(avg_corr_df.index, pd.MultiIndex):
           avg_corr_df = avg_corr_df.unstack(level=1)
           avg_corr_df.columns = [f"{col[0]}" for col in avg_corr_df.columns]
    
    enriched_features_list.append(avg_corr_df)
    
    # --- รวมทุก Feature เข้าด้วยกัน ---
    final_enriched_features = pd.concat(enriched_features_list, axis=1)
    
    # เรียงคอลัมน์ตามชื่อสินทรัพย์ เพื่อให้ feature ของสินทรัพย์เดียวกันอยู่ติดกัน
    final_enriched_features = final_enriched_features.reindex(sorted(final_enriched_features.columns), axis=1)

    final_enriched_features.fillna(method='ffill', inplace=True)
    final_enriched_features.fillna(0, inplace=True)
    
    print(f"MPT-Enriched features created with shape: {final_enriched_features.shape}")
    return final_enriched_features

def create_mpt_enriched_features_hrc(price_data, lookback_window=63):
    print("Creating MPT-Enriched features for Hierarchical Model...")
    enriched_features_list = []
    returns = price_data.pct_change()

    # ส่วนที่ 1: มิติผลตอบแทน (Return Component)
    print("  - Calculating Return/Momentum features...")
    for n in [5, 21, 63]:
        enriched_features_list.append(returns.rolling(window=n, min_periods=1).mean().add_suffix(f'_ret_{n}d'))

    # ส่วนที่ 2: มิติความเสี่ยงเฉพาะตัว (Individual Risk Component)
    print("  - Calculating Risk/Volatility features...")
    for n in [21, 63]:
        enriched_features_list.append(returns.rolling(window=n, min_periods=1).std().add_suffix(f'_vol_{n}d'))

    # ส่วนที่ 3: มิติความสัมพันธ์ (Correlation/Diversification Component)
    print("  - Calculating Correlation features...")
    corr_matrix_rolling = returns.rolling(window=lookback_window, min_periods=lookback_window//2).corr(pairwise=True)
    avg_corr_data = {}
    for asset in price_data.columns:
        if asset in corr_matrix_rolling.index.get_level_values(1):
             avg_corr = corr_matrix_rolling.xs(asset, level=1).drop(columns=asset, errors='ignore').mean(axis=1)
             avg_corr_data[f'{asset}_avg_corr'] = avg_corr
    avg_corr_df = pd.DataFrame(avg_corr_data)
    if not avg_corr_df.empty and isinstance(avg_corr_df.index, pd.MultiIndex):
       avg_corr_df = avg_corr_df.unstack(level=1)
       avg_corr_df.columns = [f"{col[0]}" for col in avg_corr_df.columns]
    enriched_features_list.append(avg_corr_df)
    
    final_features = pd.concat(enriched_features_list, axis=1)
    final_features = final_features.reindex(sorted(final_features.columns), axis=1)
    final_features.fillna(method='ffill', inplace=True)
    final_features.fillna(0, inplace=True)
    
    print(f"MPT-Enriched features created with shape: {final_features.shape}")
    return final_features

def create_adjacency_matrix(asset_map, asset_list):
    """
    สร้าง Adjacency Matrix โดยให้สินทรัพย์ที่อยู่ในกลุ่มเดียวกัน (sector) มีเส้นเชื่อมถึงกัน
    """
    num_assets = len(asset_list)
    adj_matrix = np.zeros((num_assets, num_assets))
    
    asset_to_idx = {asset: i for i, asset in enumerate(asset_list)}
    
    for i in range(num_assets):
        for j in range(i, num_assets):
            asset1 = asset_list[i]
            asset2 = asset_list[j]
            # ให้มีเส้นเชื่อมถ้าอยู่ในกลุ่มเดียวกัน หรือถ้าเป็นสินทรัพย์ตัวเดียวกัน (self-loop)
            if i == j or (asset_map.get(asset1) == asset_map.get(asset2)):
                adj_matrix[i, j] = 1
                adj_matrix[j, i] = 1 # Matrix สมมาตร
                
    print("Adjacency matrix created for the graph.")
    return adj_matrix


# ===== Direct Sequence =====

# สร้างชุดข้อมูลสำหรับ Hierarchical Model
def create_hierarchical_forecast_dataset(features, prices, sequence_length, forecast_horizon):
    """
    **ใหม่**: สร้างชุดข้อมูลสำหรับ Hierarchical Model
    Output X จะมี 4 มิติ: (samples, assets, sequence, features_per_asset)
    """
    num_assets = len(prices.columns)
    # คำนวณจำนวน feature ต่อ 1 asset
    num_features_per_asset = features.shape[1] // num_assets
    
    # Reshape features จาก 2D -> 3D
    # (days, assets * features_per_asset) -> (days, assets, features_per_asset)
    feature_values_3d = features.values.reshape(len(features), num_assets, num_features_per_asset)
    price_values = prices.values
    
    x_data, y_data = [], []
    for i in range(len(feature_values_3d) - sequence_length - forecast_horizon + 1):
        # Input (X) คือ features ย้อนหลัง
        x_slice = feature_values_3d[i : i + sequence_length]
        # Transpose เพื่อให้เป็น (assets, sequence, features_per_asset)
        x_data.append(np.transpose(x_slice, (1, 0, 2)))
        
        # Target (y) คือ returns ในอนาคต (เหมือนเดิม)
        price_slice_future = price_values[i + sequence_length : i + sequence_length + forecast_horizon]
        price_slice_for_calc = np.vstack([price_values[i + sequence_length - 1], price_slice_future])
        returns_slice = (price_slice_for_calc[1:] - price_slice_for_calc[:-1]) / price_slice_for_calc[:-1]
        np.nan_to_num(returns_slice, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        y_data.append(returns_slice)
        
    return np.array(x_data), np.array(y_data)

# Cross Attention
def create_mpt_cross_features(price_data, lookback_window=63):
    """
    **ใหม่**: สร้าง Feature 2 ชุดแยกกันสำหรับ Cross-Attention ตามหลัก MPT
    1. Returns Features (สำหรับ Query)
    2. Risk Features (สำหรับ Key/Value)
    """
    print("Creating MPT-based features for Cross-Attention...")
    returns = price_data.pct_change()
    
    # --- ชุดที่ 1: Returns Features (Query) ---
    returns_features = pd.DataFrame(index=price_data.index)
    for n in [5, 21, 63]:
        returns_features = returns_features.join(returns.rolling(window=n, min_periods=1).mean().add_suffix(f'_ret_{n}d'))
    
    # --- ชุดที่ 2: Risk & Correlation Features (Key/Value) ---
    risk_features = pd.DataFrame(index=price_data.index)
    for n in [21, 63]:
        risk_features = risk_features.join(returns.rolling(window=n, min_periods=1).std().add_suffix(f'_vol_{n}d'))
    
    # เพิ่ม Proxy สำหรับ Correlation
    corr_matrix_rolling = returns.rolling(window=lookback_window, min_periods=lookback_window//2).corr(pairwise=True)
    avg_corr_data = {}
    for asset in price_data.columns:
        if asset in corr_matrix_rolling.index.get_level_values(1):
             avg_corr = corr_matrix_rolling.xs(asset, level=1).drop(columns=asset, errors='ignore').mean(axis=1)
             avg_corr_data[f'{asset}_avg_corr'] = avg_corr
    avg_corr_df = pd.DataFrame(avg_corr_data)
    if not avg_corr_df.empty and isinstance(avg_corr_df.index, pd.MultiIndex):
       avg_corr_df = avg_corr_df.unstack(level=1)
       avg_corr_df.columns = [f"{col[0]}" for col in avg_corr_df.columns]
    risk_features = risk_features.join(avg_corr_df)

    # จัดการค่าว่างและเรียงคอลัมน์
    returns_features = returns_features.reindex(sorted(returns_features.columns), axis=1).fillna(0)
    risk_features = risk_features.reindex(sorted(risk_features.columns), axis=1).fillna(0)

    print("MPT cross-features created.")
    return returns_features, risk_features

def create_cross_attention_forecast_dataset(returns_features, risk_features, prices, sequence_length, forecast_horizon):
    """
    **ใหม่**: สร้างชุดข้อมูลสำหรับ Cross-Attention Model
    Output X จะเป็น tuple หรือ list ที่มี 2 สมาชิก: (X_query, X_key_value)
    """
    x_q_data, x_kv_data, y_data = [], [], []
    q_values = returns_features.values
    kv_values = risk_features.values
    price_values = prices.values
    
    for i in range(len(q_values) - sequence_length - forecast_horizon + 1):
        x_q_data.append(q_values[i : i + sequence_length])
        x_kv_data.append(kv_values[i : i + sequence_length])
        
        # Target (y) คือ returns ในอนาคต (เหมือนเดิม)
        price_slice_future = price_values[i + sequence_length : i + sequence_length + forecast_horizon]
        price_slice_for_calc = np.vstack([price_values[i + sequence_length - 1], price_slice_future])
        returns_slice = (price_slice_for_calc[1:] - price_slice_for_calc[:-1]) / price_slice_for_calc[:-1]
        np.nan_to_num(returns_slice, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        y_data.append(returns_slice)
        
    return [np.array(x_q_data), np.array(x_kv_data)], np.array(y_data)

# GAT
def create_graph_forecast_dataset(features, prices, adj_matrix, sequence_length, forecast_horizon):
    """
    **ใหม่**: สร้างชุดข้อมูลสำหรับ Graph Model
    Output X จะเป็น tuple หรือ list ที่มี 2 สมาชิก: (X_features, X_adj)
    """
    num_assets = len(prices.columns)
    num_features_per_asset = features.shape[1] // num_assets
    
    feature_values_3d = features.values.reshape(len(features), num_assets, num_features_per_asset)
    price_values = prices.values
    
    x_features_data, y_data = [], []
    for i in range(len(feature_values_3d) - sequence_length - forecast_horizon + 1):
        x_slice = feature_values_3d[i : i + sequence_length]
        x_features_data.append(x_slice)
        
        price_slice_future = price_values[i + sequence_length : i + sequence_length + forecast_horizon]
        price_slice_for_calc = np.vstack([price_values[i + sequence_length - 1], price_slice_future])
        returns_slice = (price_slice_for_calc[1:] - price_slice_for_calc[:-1]) / price_slice_for_calc[:-1]
        np.nan_to_num(returns_slice, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        y_data.append(returns_slice)
        
    x_features_np = np.array(x_features_data)
    y_np = np.array(y_data)
    
    # Adjacency matrix จะเหมือนกันสำหรับทุก sample
    x_adj_np = np.repeat(adj_matrix[np.newaxis, ...], x_features_np.shape[0], axis=0)
    
    return [x_features_np, x_adj_np], y_np

def create_adjacency_matrix_2(asset_map, asset_list): # (ฟังก์ชันใหม่จากเวอร์ชันก่อน)
    num_assets = len(asset_list)
    adj_matrix = np.zeros((num_assets, num_assets))
    for i in range(num_assets):
        for j in range(i, num_assets):
            if i == j or (asset_map.get(asset_list[i]) == asset_map.get(asset_list[j])):
                adj_matrix[i, j] = 1
                adj_matrix[j, i] = 1
    print("Adjacency matrix created for the graph.")
    return adj_matrix


# ============ Fund Performance Analysis ============

def calculate_metrics(series, risk_free_rate=0.02):
    if len(series) < 2:
        return {
            'Total Return (%)': 0,
            'Annualized Return (%)': 0,
            'Volatility (%)': 0,
            'Sharpe Ratio': 0,
            'Max Drawdown (%)': 0,
            'Max Drawdown Duration (days)': 0,
            'Sortino Ratio': 0
        }
    
    # Calculate daily returns
    returns = series.pct_change().dropna()
    
    # Add this crucial check for empty returns
    if len(returns) == 0:
        return {
            'Total Return (%)': np.nan,
            'Annualized Return (%)': np.nan,
            'Volatility (%)': np.nan,
            'Sharpe Ratio': np.nan,
            'Max Drawdown (%)': np.nan,
            'Max Drawdown Duration (days)': np.nan,
            'Sortino Ratio': np.nan
        }

    # Total Return (%)
    total_return = (1 + returns).prod() - 1
    
    annualized_return = (1 + returns).prod() ** (252 / len(returns)) - 1
    volatility = returns.std() * np.sqrt(252)
    
    # Max drawdown
    max_dd, max_dd_duration = calculate_max_drawdown(returns)

    # Risk-adjusted metrics
    excess_returns = returns - risk_free_rate/252
    sharpe_ratio = excess_returns.mean() / returns.std() * np.sqrt(252)
    
    # Sortino ratio (downside deviation)
    downside_returns = returns[returns < 0]
    sortino_ratio = excess_returns.mean() / downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else np.nan
    
    # Max drawdown for the quarter
    if len(returns) > 0:
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_dd = drawdown.min()
    else:
        max_dd = np.nan

    return {
        'Total Return (%)': total_return*100,
        'Annualized Return (%)': annualized_return*100,
        'Volatility (%)': volatility*100,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown (%)': max_dd*100,
        'Max Drawdown Duration (days)': max_dd_duration,
        'Sortino Ratio': sortino_ratio
    }

def fund_performance_analysis(df):
    # 1. All Period (2020-2024)
    all_period_results = {}
    for fund in df.columns:
        all_period_results[fund] = calculate_metrics(df[fund])
    df_all_period = pd.DataFrame(all_period_results).T

    # 2. Yearly breakdown
    years = df.index.year.unique()
    yearly_results = []

    for year in years:
        df_year = df[df.index.year == year]
        if len(df_year) < 2:
            continue
            
        for fund in df.columns:
            metrics = calculate_metrics(df_year[fund])
            metrics['Year'] = year
            metrics['Fund'] = fund
            yearly_results.append(metrics)

    df_yearly = pd.DataFrame(yearly_results)
    df_yearly = df_yearly.set_index(['Year', 'Fund'])

    # 3. Quarterly breakdown
    def get_quarter(date):
        quarter = (date.month - 1) // 3 + 1
        return f"{quarter}Q{date.year}"

    df_quarter = df.copy()
    df_quarter['Quarter'] = df_quarter.index.map(get_quarter)
    quarterly_results = []

    for quarter, group in df_quarter.groupby('Quarter'):
        group = group.drop(columns='Quarter')
        if len(group) < 2:
            continue
            
        for fund in group.columns:
            metrics = calculate_metrics(group[fund])
            metrics['Quarter'] = quarter
            metrics['Fund'] = fund
            quarterly_results.append(metrics)

    df_quarterly = pd.DataFrame(quarterly_results)
    df_quarterly = df_quarterly.set_index(['Quarter', 'Fund'])

    return df_all_period, df_yearly, df_quarterly

def calculate_peer_percentiles(df, groupby=None):
    """
    คำนวณ Peer Percentiles (5th, 25th, 50th, 75th, 95th) 
    หากระบุ groupby เช่น 'Year' หรือ 'Quarter' จะคำนวณในแต่ละกลุ่ม
    """
    percentiles = [0.05, 0.25, 0.5, 0.75, 0.95]

    if groupby:
        return df.groupby(level=groupby).quantile(percentiles).unstack(level=-1)
    else:
        return df.quantile(percentiles).T


def mvo_quarterly_no_sec_cons(data, port_type, 
                          start_year=2020, trading_days_per_quarter=63, min_train_periods=252, risk_free_rate=0.02):
    weights_dict = {}
    asset_symbols = list(data.columns)
    
    rebalance_dates = get_rebalance_dates(data, start_year)
    # Store original datetime index
    original_index = data.index.copy()
    # Reset index to integer for processing
    df_reset = data.copy()
    df_reset.index = range(len(df_reset))

    for i, rebalance_date in enumerate(rebalance_dates):
        print(f"\n--- Rebalancing {i+1}/{len(rebalance_dates)} ---")
        print(f"Rebalance Date: {rebalance_date.strftime('%Y-%m-%d')}")
        
        # Find corresponding position in reset dataframe
        try:
            train_end_pos = original_index.get_loc(rebalance_date) - 1
        except KeyError:
            # Find nearest date if exact date doesn't exist
            nearest_dates = original_index[original_index >= rebalance_date]
            if len(nearest_dates) == 0:
                print(f"No data available for {rebalance_date}, skipping...")
                continue
            train_end_pos = original_index.get_loc(nearest_dates[0])
        
        # Define training period - 1 day before rebalance date
        train_start_pos = max(0, train_end_pos - min_train_periods)
        
        print(f"Training period: {original_index[train_start_pos].strftime('%Y-%m-%d')} to {original_index[train_end_pos].strftime('%Y-%m-%d')}")
        print(f"Training days: {train_end_pos - train_start_pos + 1}")
        
        # Get quarter end date
        quarter_end = get_quarter_end_date(rebalance_date, original_index, trading_days_per_quarter)
        
        # Extract training data
        train_data = df_reset.iloc[train_start_pos:train_end_pos+1].copy()
        
        train_data = train_data.ffill().bfill()
        
        lookback_data = train_data.copy()
        if len(lookback_data) < 50:  # Need sufficient data
            print(f"❌ Insufficient data for {rebalance_date}")
            continue
            
        try:
            mu = expected_returns.mean_historical_return(lookback_data)
            S = risk_models.sample_cov(lookback_data)
        except Exception as e:
            print(f"❌ Could not calculate returns/covariance: {e}")
            continue

        # Set up the optimizer
        ef_agg = EfficientFrontier(mu, S, weight_bounds=port_type)
        # ef_agg.add_sector_constraints(asset_map, asset_lower, asset_upper)
        ef_agg.add_constraint(lambda x: x <= 0.3)

        # Find the portfolio that maximizes the Sharpe ratio
        try:
            weights_agg = ef_agg.max_sharpe(risk_free_rate=risk_free_rate)
            cleaned_weights_agg = ef_agg.clean_weights()
        except Exception as e:
            print(f"❌ Optimization failed for {rebalance_date.strftime('%Y-%m-%d')}: {e}")
            continue
        
        # Store weights for this rebalance date
        weights_dict[rebalance_date] = cleaned_weights_agg
    
    # Create DataFrame with rebalance_dates as index and ETF symbols as columns
    optimal_weights_df = pd.DataFrame.from_dict(
        weights_dict, 
        orient='index'
    )
    # Ensure all original assets are columns, filling missing with 0
    optimal_weights_df = optimal_weights_df.reindex(columns=data.columns, fill_value=0)
    
    return optimal_weights_df