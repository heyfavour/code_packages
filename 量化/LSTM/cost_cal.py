import sys,os
def get_deal_cost(buy_price,sell_price,nums):
    sell = (sell_price*100*nums - sell_price*100*0.001*nums - max(sell_price*100*2.5*0.0001*nums,5))
    buy = (buy_price*100*nums+max(buy_price*100*2.5*0.0001*nums,5))
    return round(sell- buy,2)

if __name__ == '__main__':
    buy_price = 34.84
    sell_price = buy_price + 0.06
    nums = 6
    delta = get_deal_cost(buy_price,sell_price,nums)
    data = {
        "increase":round((sell_price-buy_price)*100/buy_price,2),
        "buy_price":buy_price,
        "sell_price":round(sell_price,2),
        "nums":nums*100,
        "delta":delta,
    }
    msg = """涨幅: {increase}\n买价：{buy_price}\n卖价：{sell_price}\n交易数量：{nums}\n收益:{delta}""".format(**data)
    print(msg)