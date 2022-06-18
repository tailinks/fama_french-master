def net_net_calculator(total_current_assets: float, total_liabilities: float,
                       preferred_stock: float) -> float:
    net_net = total_current_assets - total_liabilities - preferred_stock
    return net_net


def net_net_ratio_calculator(net_net: float, market_cap: float) -> float:
    net_net_ratio = net_net / market_cap
    return net_net_ratio


def earnings_to_price_calculator(market_cap: float, earnings: float) -> float:
    earnings_to_price = earnings / market_cap
    return earnings_to_price


def book_value_calculator(total_assets: float, total_liabilities: float,
                          preferred_stock: float, ) -> float:
    book_value = total_assets - total_liabilities - preferred_stock
    return book_value


def book_value_to_price_calculator(book_value: float, market_cap: float) -> float:
    book_to_price = book_value / market_cap
    return book_to_price


def price_to_free_cash_flow_calculator(market_cap: float, free_cash_flow: float) -> float:
    price_to_free_cash_flow = market_cap / free_cash_flow
    return price_to_free_cash_flow


def price_to_sales_calculator(market_cap: float, sales: float) -> float:
    price_to_sales = market_cap / sales
    return price_to_sales


def average_earnings_to_price_calculator(market_cap: float, *earnings) -> float:
    average_earnings = sum(earnings) / len(earnings)
    average_earnings_to_price = average_earnings / market_cap
    return average_earnings_to_price
