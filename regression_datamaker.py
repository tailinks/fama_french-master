import numpy as np
import pandas as pd
import math
from fmp import *

pd.options.mode.chained_assignment = None


def find_most_recent_dict(list_of_dict_to_transform: list, list_of_dict_to_add: list, to_match_key='start',
                          matching_key='date') -> list:
    def nearest(items, pivot, key=matching_key):
        return min([i for i in items if i[key] <= pivot], key=lambda x: abs(x[key] - pivot))

    new_list = []
    used_list = list_of_dict_to_add
    for dicts_to_mod in list_of_dict_to_transform:
        to_add = nearest(used_list, dicts_to_mod[to_match_key], key=matching_key)
        result_dict = {**dicts_to_mod, **to_add}
        new_list.append(result_dict)
    return new_list


def keep_every_n_rows(df: pd.DataFrame, n: int, keep='bottom') -> pd.DataFrame:
    """

    :param df: Dataframe to be modified
    :param n: n rows to keep
    :param keep: If set to bottom will keep the last row, else it will keep the top one.
    :return:
    """
    if keep == 'bottom':
        df = df.iloc[::-1]
    df = df[::n]
    if keep == 'bottom':
        df = df.iloc[::-1]
    return df


def add_non_native_timeframe(daily_prices: pd.DataFrame, new_time_frame: str) -> pd.DataFrame:
    """

    :param daily_prices: Daily prices dataframe
    :param new_time_frame: New time frame can be monthly or weekly
    :return: Returns the new_time_frame dataframe
    """
    working_df = daily_prices.copy()
    working_df['date'] = working_df.index
    working_df['date'] = pd.to_datetime(working_df['date'])
    if new_time_frame == "week":
        working_df[new_time_frame] = working_df['date'].dt.isocalendar().week
    elif new_time_frame == "month":
        working_df[new_time_frame] = working_df['date'].dt.month
    working_df['year'] = working_df['date'].dt.isocalendar().year
    if 'volume' in working_df.columns:
        result_df = working_df.groupby(['year', new_time_frame]).agg(
            {'date': 'first', 'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
    else:
        result_df = working_df.groupby(['year', new_time_frame]).agg(
            {'date': 'first', 'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'})
    result_df['date'] = result_df['date'].dt.date
    result_df.set_index('date', inplace=True)
    return result_df


def dividends_sum(start: datetime.date, end: datetime.date, dividend_list: list) -> float:
    div_sum = 0
    for item in dividend_list:
        if start < item['date'] <= end:
            try:
                div_sum += item['dividend']
            except KeyError:
                div_sum += item['adjDividend']
    return div_sum


def quarterly_income_ttm_converter(list_of_dict: list) -> list:
    columns = ['revenue', 'costOfRevenue',
               'grossProfit', 'grossProfitRatio', 'researchAndDevelopmentExpenses',
               'generalAndAdministrativeExpenses', 'sellingAndMarketingExpenses',
               'sellingGeneralAndAdministrativeExpenses', 'otherExpenses',
               'operatingExpenses', 'costAndExpenses', 'interestIncome',
               'interestExpense', 'depreciationAndAmortization', 'ebitda',
               'ebitdaratio', 'operatingIncome', 'operatingIncomeRatio',
               'totalOtherIncomeExpensesNet', 'incomeBeforeTax',
               'incomeBeforeTaxRatio', 'incomeTaxExpense', 'netIncome',
               'netIncomeRatio', 'eps', 'epsdiluted']
    quarterly_income_df = pd.DataFrame(list_of_dict)
    quarterly_income_df = quarterly_income_df.iloc[::-1]
    quarterly_income_df[columns] = quarterly_income_df[columns].rolling(4).sum()
    quarterly_income_df.dropna(inplace=True)
    quarterly_income_ttm = quarterly_income_df.to_dict('records')
    return quarterly_income_ttm


def quarterly_cash_flow_ttm_converter(list_of_dict: list) -> list:
    columns = ['netIncome',
               'depreciationAndAmortization', 'deferredIncomeTax',
               'stockBasedCompensation', 'changeInWorkingCapital',
               'accountsReceivables', 'inventory', 'accountsPayables',
               'otherWorkingCapital', 'otherNonCashItems',
               'netCashProvidedByOperatingActivities',
               'investmentsInPropertyPlantAndEquipment', 'acquisitionsNet',
               'purchasesOfInvestments', 'salesMaturitiesOfInvestments',
               'otherInvestingActivites', 'netCashUsedForInvestingActivites',
               'debtRepayment', 'commonStockIssued', 'commonStockRepurchased',
               'dividendsPaid', 'otherFinancingActivites',
               'netCashUsedProvidedByFinancingActivities',
               'effectOfForexChangesOnCash', 'netChangeInCash', 'cashAtEndOfPeriod',
               'cashAtBeginningOfPeriod', 'operatingCashFlow', 'capitalExpenditure',
               'freeCashFlow']
    quarterly_cash_flow_df = pd.DataFrame(list_of_dict)
    quarterly_cash_flow_df = quarterly_cash_flow_df.iloc[::-1]
    quarterly_cash_flow_df[columns] = quarterly_cash_flow_df[columns].rolling(4).sum()
    quarterly_cash_flow_df.dropna(inplace=True)
    quarterly_cash_flow_ttm = quarterly_cash_flow_df.to_dict('records')
    return quarterly_cash_flow_ttm


def change_calculator(latest_row: float, earliest_row: float) -> float:
    """

    :param latest_row: Latest data
    :param earliest_row: Earliest data
    :return: % change between two
    """
    change = (latest_row - earliest_row) / earliest_row
    return change


def returns_calculator(historical_prices_df: pd.DataFrame, dividends_obj: list, length=1) -> pd.DataFrame:
    returns_df = historical_prices_df[['close']]
    returns_df['end'] = returns_df.index
    returns_df['start'] = returns_df['end'].shift(1)
    returns_dicts_list = returns_df.dropna().to_dict('records')
    if not dividends_obj:
        returns_df = pd.DataFrame(returns_dicts_list)
        returns_df['%_change'] = np.vectorize(change_calculator)(returns_df['close'],
                                                                 returns_df['close'].shift(1))
        return returns_df.set_index('end')[['%_change']]
    else:
        for values in returns_dicts_list:
            values['dividends'] = dividends_sum(values['start'], values['end'], dividends_obj)
        returns_df = pd.DataFrame(returns_dicts_list)
        returns_df['close_w_dividends'] = returns_df['close'] + returns_df['dividends']
        returns_df['%_change'] = np.vectorize(change_calculator)(returns_df['close_w_dividends'],
                                                                 returns_df['close'].shift(1))
        returns_df = keep_every_n_rows(returns_df, length)
        return returns_df.set_index('end')[['%_change']]


def excess_returns_calculator(asset_returns_df: pd.DataFrame, benchmark_returns_df: pd.DataFrame) -> pd.DataFrame:
    excess_return_df = pd.concat([asset_returns_df, benchmark_returns_df], axis=1)
    excess_return_df['excess_return'] = excess_return_df['%_change'] - excess_return_df['benchmark_%_change']
    return excess_return_df


def get_clean_daily_prices(symbol: str):
    raw_daily_prices = pd.DataFrame(get_historical_prices(symbol, 'daily'))[::-1].reset_index(drop=True)
    clean_daily_prices = raw_daily_prices.set_index('date').drop(
        columns=['adjClose', 'change', 'changePercent', 'vwap',
                 'label', 'changeOverTime', 'unadjustedVolume'], errors='ignore')
    return clean_daily_prices


def get_average_earnings(earnings: pd.Series, periods: int) -> pd.Series:
    average_earnings = earnings.rolling(window=periods).mean()
    return average_earnings


class treasury_rates:

    def __init__(self):
        current = date_50y_ago
        end = date_today
        delta = relativedelta(months=+3)
        rates = []
        while current <= end + delta:
            current_rates = get_treasury_rates(current, current + delta)
            date_modifier(current_rates, 'date')
            for rate in current_rates:
                rates.append(rate)
            current += delta

        raw_rates = pd.DataFrame(rates).drop_duplicates().dropna(how='all').sort_index()
        self.rates = raw_rates.to_dict('records')


class asset_data(object):
    def __init__(self, symbol: str, is_ETF=False) -> None:
        """
        :param symbol: Symbol to get prices for.
        """
        self.symbol = symbol
        self.daily_prices = get_clean_daily_prices(symbol)
        self.weekly_prices = add_non_native_timeframe(self.daily_prices, 'week')
        self.monthly_prices = add_non_native_timeframe(self.daily_prices, 'month')
        self.historical_dividends = get_historical_dividends(self.symbol)
        self.time_frame_selector = {"daily": self.daily_prices, "weekly": self.weekly_prices,
                                    "monthly": self.monthly_prices}
        self.market_caps = get_historical_market_caps(self.symbol)
        if not is_ETF:
            self.quarterly_balance_sheets = get_balance_sheets(self.symbol, period='quarter')
            self.quarterly_balance_sheets.reverse()
            self.quarterly_cash_flow_statements = get_cash_flow_statement(self.symbol, period='quarter')
            self.quarterly_cash_flow_ttm = quarterly_cash_flow_ttm_converter(self.quarterly_cash_flow_statements)
            self.quarterly_income_statements = get_income_statement(self.symbol, period='quarter')
            self.quarterly_income_ttm = quarterly_income_ttm_converter(self.quarterly_income_statements)
            '''self.balance_sheets = get_balance_sheets(self.symbol, period='annual')
            self.income_statements = get_income_statement(self.symbol, period='annual')
            self.cash_flow_statement = get_cash_flow_statement(self.symbol, period='annual')'''
            self.first_quarterly_income_statement = min(x['fillingDate'] for x in self.quarterly_income_ttm)
            self.first_quarterly_balance_sheet = min(x['fillingDate'] for x in self.quarterly_balance_sheets)
            self.first_quarterly_cash_flow = min(x['fillingDate'] for x in self.quarterly_cash_flow_ttm)
            self.first_full_financial_statement = max([self.first_quarterly_income_statement,
                                                       self.first_quarterly_balance_sheet,
                                                       self.first_quarterly_cash_flow])

    def get_returns(self, time_frame: str, length: int) -> pd.DataFrame:
        historical_prices_used = self.time_frame_selector[time_frame]
        historical_returns = returns_calculator(historical_prices_used, self.historical_dividends, length)
        return historical_returns


class regression_dataset:
    def __init__(self, symbol: str, benchmark_returns: pd.DataFrame, t_rates: list):
        self.symbol = symbol
        self.asset_data = asset_data(self.symbol)
        self.benchmark_returns = benchmark_returns.rename(columns={'%_change': 'benchmark_%_change'})
        self.t_rates = t_rates
        self.ratios = get_financial_ratios(self.symbol)

    def excess_return_data_set_maker(self, time_frame: str, length: int) -> pd.DataFrame:
        asset_return_df = self.asset_data.get_returns(time_frame=time_frame, length=length)
        excess_returns_df = excess_returns_calculator(asset_return_df, self.benchmark_returns)
        excess_returns_df.reset_index(drop=False, inplace=True)
        excess_returns_df['start'] = excess_returns_df['end'].shift(1)
        excess_returns_df = excess_returns_df[excess_returns_df['start'] >=
                                              self.asset_data.first_full_financial_statement]
        excess_returns_list_of_dict_data_set = excess_returns_df.dropna().to_dict('records')
        excess_returns_list_of_dict_data_set = find_most_recent_dict(excess_returns_list_of_dict_data_set,
                                                                     self.asset_data.quarterly_income_ttm,
                                                                     matching_key='fillingDate')
        excess_returns_list_of_dict_data_set = find_most_recent_dict(excess_returns_list_of_dict_data_set, self.ratios)
        excess_returns_list_of_dict_data_set = find_most_recent_dict(excess_returns_list_of_dict_data_set,
                                                                     self.asset_data.quarterly_cash_flow_ttm,
                                                                     matching_key='fillingDate')
        excess_returns_list_of_dict_data_set = find_most_recent_dict(excess_returns_list_of_dict_data_set,
                                                                     self.asset_data.quarterly_balance_sheets,
                                                                     matching_key='fillingDate')
        excess_returns_list_of_dict_data_set = find_most_recent_dict(excess_returns_list_of_dict_data_set,
                                                                     self.asset_data.market_caps)
        excess_returns_list_of_dict_data_set = find_most_recent_dict(excess_returns_list_of_dict_data_set,
                                                                     self.t_rates)
        regression_dataset_df = pd.DataFrame(excess_returns_list_of_dict_data_set)
        regression_dataset_df['book_value'] = \
            np.vectorize(book_value_calculator) \
                (regression_dataset_df['totalAssets'],
                 regression_dataset_df['totalLiabilities'],
                 regression_dataset_df['preferredStock'])
        regression_dataset_df['book_value_ratio'] = np.vectorize(book_value_to_price_calculator) \
            (regression_dataset_df['book_value'], regression_dataset_df['marketCap'])
        regression_dataset_df['3_y_earnings'] = get_average_earnings(regression_dataset_df['netIncome'], 3)
        regression_dataset_df['average_3_y_p/e'] = np.vectorize(earnings_to_price_calculator) \
            (regression_dataset_df['marketCap'], regression_dataset_df['3_y_earnings'])
        regression_dataset_df['log_marketCap'] = np.vectorize(math.log)(regression_dataset_df['marketCap'])
        regression_dataset_df['p/e'] = np.vectorize(earnings_to_price_calculator)(regression_dataset_df['marketCap'],
                                                                                  regression_dataset_df['netIncome'])
        regression_dataset_df['net_net'] = np.vectorize(net_net_calculator)(regression_dataset_df['totalCurrentAssets'],
                                                                            regression_dataset_df['totalLiabilities'],
                                                                            regression_dataset_df['preferredStock'])
        regression_dataset_df['net_net_ratio'] = np.vectorize(net_net_ratio_calculator)(
            regression_dataset_df['net_net'], regression_dataset_df['marketCap'])

        regression_dataset_df['last_%_change'] = regression_dataset_df['%_change'].shift(1)
        regression_dataset_df['last_bench_change'] = regression_dataset_df['benchmark_%_change'].shift(1)
        regression_dataset_df['last_excess'] = regression_dataset_df['excess_return'].shift(1)
        return regression_dataset_df
