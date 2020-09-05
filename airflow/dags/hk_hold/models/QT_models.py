# coding: utf-8
from sqlalchemy import Column, Float, String, Text,DECIMAL,Integer
from sqlalchemy.dialects.mysql import BIGINT
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()
metadata = Base.metadata


class HK_HOLD(Base):
    __tablename__ = 'hk_hold'

    HDDATE = Column(Integer, primary_key=True, nullable=False,comment = '日期')
    HKCODE = Column(String(16), primary_key=True, nullable=False,comment='HK代码')
    SCODE = Column(String(16),comment="A股代码")
    SNAME = Column(String(128),comment = 'A股名称')
    SHAREHOLDSUM = Column(Float(asdecimal=True),comment = "持股量")
    SHARESRATE = Column(DECIMAL(10,4),comment = "持股比例")
    CLOSEPRICE = Column(DECIMAL(10,4),comment="收盘价")
    ZDF = Column(Float(asdecimal=True),doc="涨跌幅")
    SHAREHOLDPRICE = Column(Float(asdecimal=True),comment="持股市值")
    SHAREHOLDPRICEONE = Column(Float(asdecimal=True),comment="持股市值一日变化")
    SHAREHOLDPRICEFIVE = Column(Float(asdecimal=True),comment="持股市值五日变化")
    SHAREHOLDPRICETEN = Column(Float(asdecimal=True),comment="持股市值十日变化")
    MARKET = Column(String(16),comment="市场")
    ShareHoldSumChg = Column(Float(asdecimal=True),comment="未知")
    Zb = Column(Text,comment="未知")
    Zzb = Column(Text,comment="未知")

    def columns(self):
        return [c.name for c in self.__table__.columns]


