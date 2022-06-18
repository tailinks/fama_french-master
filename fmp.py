import datetime
import json
from urllib.request import urlopen
from dateutil.relativedelta import relativedelta
import time
import urllib.parse

base_url = "https://financialmodelingprep.com/api/"
from config import api_key

date_today = datetime.date.today()
date_25y_ago = date_today - relativedelta(years=+25)


def get_json_parsed_data(url: str, sleep: float) -> list:
    """
    :param url: URL with json file.
    :param sleep: Time to sleep for to prevent API limit.
    :return: Dict containing parsed json data.
    """
    response = urlopen(url)
    time.sleep(sleep)
    data = response.read().decode("utf-8")
    return json.loads(data)


def ends_with_list_dropper(target_list: list, args_to_remove: list) -> list:
    """

    :param target_list: List of strings with strings to remove
    :param args_to_remove: List of ends with to remove
    :return: Proper list with unwanted strings removed
    """
    for arg in args_to_remove:
        target_list = [val for val in target_list if not val.endswith(arg)]
    return target_list


def download_ticker_list(*args) -> list:
    """

    :param args: All ends with to remove from downloaded list. Should be specific exchange ending.
    :return:
    """
    url = base_url + "v3/financial-statement-symbol-lists?apikey=" + api_key
    ticker_list = get_json_parsed_data(url, 0.081)
    # Removes unneeded tickers from specified exchanges
    if len(args) >= 1:
        ticker_list = ends_with_list_dropper(ticker_list, list(args))
    return ticker_list


def date_modifier(target_list: list, key: str, date_format="%Y-%m-%d") -> None:
    """
    :param target_list: List of dicts with a date as string each.
    :param key: Key of date.
    :param date_format: Format of the date string.
    :return:
    """
    for item in target_list:
        item[key] = datetime.datetime.strptime(item[key], date_format).date()


def get_balance_sheets(symbol: str, period='quarter') -> list:
    """
    :param symbol: Symbol to get balance sheets for.
    :param period: quarter or yearly
    :return: List of balance sheets as dicts from nearliest to oldest.
    """
    balance_sheet_url = base_url + "v3/balance-sheet-statement/" + symbol + \
                        "?period=" + period + "&limit=1000&apikey=" + api_key
    balance_sheets = get_json_parsed_data(balance_sheet_url, 0.081)
    date_modifier(balance_sheets, "fillingDate")

    return balance_sheets


def get_income_statement(symbol: str, period='quarter') -> list:
    """
    :param symbol: Symbol to get income statements for.
    :param period: quarter or yearly
    :return: List of income statements as dicts from nearliest to oldest.
    """
    income_statement_url = base_url + "v3/income-statement/" + symbol + \
                           "?period=" + period + "&limit=1000&apikey=" + api_key
    income_statement = get_json_parsed_data(income_statement_url, 0.081)
    date_modifier(income_statement, "fillingDate")

    return income_statement


def get_cash_flow_statement(symbol: str, period='quarter') -> list:
    """
    :param symbol: Symbol to get cash flows for.
    :param period: quarter or yearly
    :return: List of cash flows as dicts from nearliest to oldest.
    """
    cash_flow_statement_url = base_url + "v3/cash-flow-statement/" + symbol + \
                              "?period=" + period + "&limit=1000&apikey=" + api_key
    cash_flow_statement = get_json_parsed_data(cash_flow_statement_url, 0.081)
    date_modifier(cash_flow_statement, "fillingDate")

    return cash_flow_statement


def get_company_profile(symbol: str) -> dict:
    """
    :param symbol: Symbol to get profile for.
    :return: Dict of profile
    """
    company_profile_url = base_url + "v3/profile/" + symbol + "?apikey=" + api_key
    company_profile = get_json_parsed_data(company_profile_url, 0.081)[0]
    return company_profile


def get_historical_market_caps(symbol: str) -> list:
    """

    :param symbol: Symbol to get market caps for for
    :return: List of historical market caps from nearliest to oldest.
    """
    historical_market_cap_url = base_url + "v3/historical-market-capitalization/" + symbol + \
                                "?limit=100000&apikey=" + api_key
    historical_market_caps = get_json_parsed_data(historical_market_cap_url, 0.081)
    date_modifier(historical_market_caps, 'date')

    return historical_market_caps


def get_historical_prices(symbol: str, time_frame: str) -> list:
    """

    :param time_frame: week/daily/4hour/1hour/30min/15min/5min/1min
    :param symbol: Symbol to get market prices for for
    :return: List of historical market prices at close from nearliest to oldest.
    """
    if symbol.startswith('^'):
        symbol = symbol.replace("^", "%5E")

    if time_frame == 'daily':
        url = base_url + "v3/historical-price-full/" + symbol + "?from=" + \
              str(date_25y_ago) + "&to=" + str(date_today) + "&apikey=" + api_key
    else:
        url = base_url + "v3/historical-chart/" + time_frame + "/" + symbol + "?from=" + \
              str(date_25y_ago) + "&to=" + str(date_today) + "&apikey=" + api_key
    historical_prices = get_json_parsed_data(url, 0.081)
    if time_frame == 'daily':
        historical_prices = historical_prices['historical']
        date_modifier(historical_prices, 'date')
    else:
        date_modifier(historical_prices, 'date', date_format="%Y-%m-%d %H:%M:%S")
    return historical_prices


def get_historical_dividends(symbol: str, ) -> list:
    """

    :param symbol: Symbol to get historical dividends for
    :return: List of all historical dividends
    """
    url = base_url + "v3/historical-price-full/stock_dividend/" + symbol + "?apikey=" + api_key
    historical_dividends = get_json_parsed_data(url, 0.081)
    if historical_dividends:
        historical_dividends = historical_dividends['historical']
        date_modifier(historical_dividends, 'date', date_format="%Y-%m-%d")
        return historical_dividends
    else:
        return None


def get_treasury_rates(start: str or datetime, end: str or datetime) -> list:
    url = base_url + "v4/treasury?"
    params = {'from': start, "to": end, "apikey": api_key}

    final_url = url + urllib.parse.urlencode(params)
    t_rates = get_json_parsed_data(final_url, 0.081)
    return t_rates


def get_financial_ratios(symbol: str, ) -> list:
    url = base_url + "v3/ratios/" + symbol + "?"
    params = {"period": "quarter", "limit": 10000, "apikey": api_key}
    final_url = url + urllib.parse.urlencode(params)
    ratios = get_json_parsed_data(final_url, 0.081)
    ratios.reverse()
    date_modifier(ratios, 'date', date_format="%Y-%m-%d")
    return ratios


def get_sp500_constituent(tickers_only=False) -> list:
    url = base_url + "v3/sp500_constituent?"
    params = {"apikey": api_key}
    final_url = url + urllib.parse.urlencode(params)
    sp500_constituent = get_json_parsed_data(final_url, 0.081)
    if tickers_only:
        tickers = []
        for company in sp500_constituent:
            tickers.append(company["symbol"])
        return tickers
    return sp500_constituent


def historical_sp500_constituent_modification() -> list:
    url = base_url + "v3/historical/sp500_constituent?"
    params = {"apikey": api_key}
    final_url = url + urllib.parse.urlencode(params)
    sp500_modifications = get_json_parsed_data(final_url, 0.081)
    date_modifier(sp500_modifications, 'date', date_format="%Y-%m-%d")
    return sp500_modifications


def get_dated_sp500_constituent() -> dict:
    sp500_changes = historical_sp500_constituent_modification()
    sp_500_dated_constituent = {date_today: get_sp500_constituent(tickers_only=True)}
    changes = {}

    for change in sp500_changes:
        if change['date'] not in changes:
            changes[change['date']] = {"added_securities": [], "removed_securities": []}
        if change["addedSecurity"] != "":
            changes[change['date']]["added_securities"].append(change["symbol"])
        else:
            changes[change['date']]["removed_securities"].append(change["symbol"])
    changes = {k: v for k, v in changes.items() if v is not None}
    for change in changes.items():
        modded_list = list(sp_500_dated_constituent.values())[-1]
        modded_list.extend(change[1]["removed_securities"])
        modded_list = [ticker for ticker in modded_list if ticker not in change[1]["added_securities"]]
        sp_500_dated_constituent[change[0]] = modded_list
    return sp_500_dated_constituent


def stock_screener(marketCapMoreThan="", marketCapLowerThan="", priceMoreThan="",
                   priceLowerThan="", betaMoreThan="", betaLowerThan="", volumeMoreThan="",
                   volumeLowerThan="", dividendMoreThan="", dividendLowerThan="", isEtf=False,
                   isActivelyTrading=True, sector='', Industry='', Country='', exchange='', limit=100000,
                   return_tickers_only=True):
    url = base_url + "v3/stock-screener?"
    params = {'marketCapMoreThan': marketCapMoreThan, 'marketCapLowerThan': marketCapLowerThan,
              'priceMoreThan': priceMoreThan, "priceLowerThan": priceLowerThan,
              "betaMoreThan": betaMoreThan, "betaLowerThan": betaLowerThan, "volumeMoreThan": volumeMoreThan,
              "volumeLowerThan": volumeLowerThan, "dividendMoreThan": dividendMoreThan,
              "dividendLowerThan": dividendLowerThan, "isEtf": isEtf, "isActivelyTrading": isActivelyTrading,
              "sector": sector,
              "industry": Industry, "country": Country, "exchange": exchange, "limit": limit, "apikey": api_key}
    final_url = url + urllib.parse.urlencode(params)
    api_response = get_json_parsed_data(final_url, 0.081)
    if return_tickers_only:
        tickers = [x['symbol'] for x in api_response]
        return tickers
    else:
        return api_response
