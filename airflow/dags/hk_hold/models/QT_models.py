# coding: utf-8
from sqlalchemy import Column, Float, String, DECIMAL, Integer
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()
metadata = Base.metadata


class HK_HOLD(Base):
    __tablename__ = 'hk_hold'

    date_id = Column(Integer, primary_key=True, nullable=False, comment='持股日期')
    code = Column(String(16), primary_key=True, nullable=False, comment='股票代码')
    name = Column(String(128), comment='股票简称')
    price = Column(DECIMAL(10, 4), comment="当日收盘价")
    amplitude = Column(Float(asdecimal=True), doc="当日涨跌幅")
    numbers = Column(Float(asdecimal=True), doc="持股数量")
    market_value = Column(Float(asdecimal=True), comment="持股市值")
    share_percent = Column(Float(asdecimal=True), comment="持股数量占发行股百分比")
    change_one_days = Column(Float(asdecimal=True), comment="持股市值变化-1日")
    change_five_days = Column(Float(asdecimal=True), comment="持股市值变化-5日")
    change_ten_days = Column(Float(asdecimal=True), comment="持股市值变化-10日")

    def columns(self):
        return [c.name for c in self.__table__.columns]




class SHARE_TRADE_DATE(Base):
    __tablename__ = 'share_trade_date'

    trade_date = Column(Integer, primary_key=True, nullable=False, comment='交易日')
