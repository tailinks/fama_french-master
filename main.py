import pandas as pd
from fmp import *
import numpy as np
import statsmodels.api as sm
from ratios import *

pd.options.mode.chained_assignment = None


def get_clean_daily_prices(symbol: str):
    raw_daily_prices = pd.DataFrame(get_historical_prices(symbol, 'daily'))[::-1].reset_index(drop=True)
    clean_daily_prices = raw_daily_prices.set_index('date').drop(
        columns=['adjClose', 'change', 'changePercent', 'vwap',
                 'label', 'changeOverTime', 'unadjustedVolume'], errors='ignore')
    return clean_daily_prices


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


def create_fama_french_df(data_csv='Fama_French_5_factors.csv') -> pd.DataFrame:
    fama_french_factors = pd.read_csv(data_csv)
    fama_french_factors.rename(columns={"Unnamed: 0": "date"}, inplace=True)
    fama_french_factors['date'] = pd.to_datetime(fama_french_factors['date'], format='%Y%m')
    fama_french_factors['month'] = fama_french_factors['date'].dt.month
    fama_french_factors['year'] = fama_french_factors['date'].dt.year
    fama_french_factors.set_index(['year', 'month'], inplace=True)
    fama_french_factors.drop(columns='date', inplace=True)
    return fama_french_factors


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
    result_df = result_df.reset_index().set_index('date')
    return result_df


def change_calculator(latest_row: float, earliest_row: float) -> float:
    """

    :param latest_row: Latest data
    :param earliest_row: Earliest data
    :return: % change between two
    """
    change = (latest_row - earliest_row) / earliest_row
    return change


def dividends_sum(start: datetime.date, end: datetime.date, dividend_list: list) -> float:
    div_sum = 0
    for item in dividend_list:
        if start < item['date'] <= end:
            try:
                div_sum += item['dividend']
            except KeyError:
                div_sum += item['adjDividend']
    return div_sum


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
                                                                 returns_df['close'].shift(1)) * 100
        returns_df = keep_every_n_rows(returns_df, length)
        return returns_df.set_index('end')[['%_change']]


def get_returns(historical_prices: pd.DataFrame, historical_dividends: pd.DataFrame, length: int) -> pd.DataFrame:
    historical_prices_used = historical_prices
    historical_returns = returns_calculator(historical_prices_used, historical_dividends, length)
    return historical_returns


def get_all_sp500_constituent(dated_sp500_constituent: dict) -> list:
    all_sp500_constituent = []
    for i in list(dated_sp500_constituent.values()):
        adding = [ticker for ticker in i if ticker not in all_sp500_constituent]
        all_sp500_constituent.extend(adding)
    return all_sp500_constituent


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
        self.market_caps = get_historical_market_caps(self.symbol)
        # self.daily_returns = get_returns(self.daily_prices, self.historical_dividends, 1)
        # self.weekly_return = get_returns(self.weekly_prices, self.historical_dividends, 1)
        self.monthly_return = get_returns(self.monthly_prices, self.historical_dividends, 1)

        if not is_ETF:
            self.quarterly_balance_sheets = get_balance_sheets(self.symbol, period='quarter')
            self.quarterly_balance_sheets.reverse()
            self.quarterly_cash_flow_statements = get_cash_flow_statement(self.symbol, period='quarter')
            # self.quarterly_cash_flow_ttm = quarterly_cash_flow_ttm_converter(self.quarterly_cash_flow_statements)
            self.quarterly_income_statements = get_income_statement(self.symbol, period='quarter')
            self.quarterly_income_ttm = quarterly_income_ttm_converter(self.quarterly_income_statements)

            self.first_quarterly_income_statement = min(x['fillingDate'] for x in self.quarterly_income_ttm)
            self.first_quarterly_balance_sheet = min(x['fillingDate'] for x in self.quarterly_balance_sheets)
            # self.first_quarterly_cash_flow = min(x['fillingDate'] for x in self.quarterly_cash_flow_ttm)
            self.first_full_financial_statement = max([self.first_quarterly_income_statement,
                                                       self.first_quarterly_balance_sheet])


class portfolio_dataset:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.asset_data = asset_data(self.symbol)
        self.workable_data = self.asset_data.monthly_return.copy()

    def data_set_maker(self, ) -> pd.DataFrame:
        self.workable_data.reset_index(drop=False, inplace=True)
        self.workable_data['start'] = pd.to_datetime(self.workable_data['end'].shift(1)).dt.date
        self.workable_data = self.workable_data[self.workable_data['start'] >=
                                                self.asset_data.first_full_financial_statement]
        self.workable_data = self.workable_data.dropna().to_dict('records')
        self.workable_data = find_most_recent_dict(self.workable_data,
                                                   self.asset_data.quarterly_income_ttm,
                                                   matching_key="fillingDate")
        self.workable_data = find_most_recent_dict(self.workable_data,
                                                   self.asset_data.quarterly_balance_sheets,
                                                   matching_key="fillingDate")
        self.workable_data = find_most_recent_dict(self.workable_data, self.asset_data.market_caps)
        self.workable_data = pd.DataFrame(self.workable_data)
        self.workable_data['p/e'] = np.vectorize(earnings_to_price_calculator) \
            (self.workable_data['marketCap'],
             self.workable_data['netIncome'])
        self.workable_data['book_value'] = np.vectorize(book_value_calculator)(
            self.workable_data['totalAssets'],
            self.workable_data['totalLiabilities'],
            self.workable_data['preferredStock'])
        self.workable_data['book_value_ratio'] = np.vectorize(book_value_to_price_calculator) \
            (self.workable_data['book_value'], self.workable_data['marketCap'])
        self.workable_data['last_change'] = self.workable_data["%_change"].shift(1)
        self.workable_data.set_index('start')
        self.workable_data = self.workable_data[
            ['%_change', 'marketCap', 'p/e', 'book_value_ratio', 'last_change']]


if __name__ == "__main__":
    dated_spy_constituent = get_dated_sp500_constituent()
    spy_constituent = get_all_sp500_constituent(dated_spy_constituent)
    spy_constituent_data = {}
    dates = pd.read_csv('first_trading_day_monthly')['start'].tolist()
    dates = [datetime.datetime.strptime(date, '%Y-%m-%d').date() for date in dates]

    trading_day_dated_constituent = {}
    for date in dates:
        res = dated_spy_constituent.get(date) or dated_spy_constituent[
            min(dated_spy_constituent.keys(), key=lambda key: abs(key - date))]
        trading_day_dated_constituent[date] = res

    for ticker in spy_constituent:
        print(ticker)
        try:
            dataset = portfolio_dataset(ticker)
            dataset.data_set_maker()
            spy_constituent_data[dataset.symbol] = dataset.workable_data
        except:
            pass
    dates_spy_data = {}
    for trading_month, tickers in trading_day_dated_constituent.items():
        df = pd.DataFrame(columns=['%_change', 'marketCap', 'p/e', 'book_value_ratio', 'last_change'])
        for ticker in tickers:
            try:
                selected_symbol_data = spy_constituent_data[ticker]
                selected_row = selected_symbol_data.iloc[[trading_month]]
                df.loc[ticker] = selected_row
            except ValueError:
                pass
        dates_spy_data[trading_month] = df
    for key, values in dates_spy_data.items():
        values.to_csv('data/'+str(key)+'.csv')