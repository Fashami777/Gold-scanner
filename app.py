
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timezone

st.set_page_config(page_title="Gold AI Trading Scanner", page_icon="🥇", layout="wide")

# ----------------------- Data -----------------------
@st.cache_data(ttl=60)
def get_data(ticker: str, interval: str, period: str):
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=False,
                     progress=False, threads=False)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.title)
    required = ["Open", "High", "Low", "Close"]
    if not all(c in df.columns for c in required):
        return pd.DataFrame()
    for c in required + (["Volume"] if "Volume" in df.columns else []):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "Volume" not in df.columns:
        df["Volume"] = 0.0
    df = df.dropna(subset=required)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df

def add_indicators(df):
    d = df.copy()
    c, h, l = d["Close"], d["High"], d["Low"]
    d["EMA20"] = c.ewm(span=20, adjust=False).mean()
    d["EMA50"] = c.ewm(span=50, adjust=False).mean()
    d["EMA200"] = c.ewm(span=200, adjust=False).mean()
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    d["RSI"] = 100 - (100 / (1 + rs))
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    d["MACD"] = ema12 - ema26
    d["MACDSignal"] = d["MACD"].ewm(span=9, adjust=False).mean()
    tr = pd.concat([(h-l), (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    d["ATR"] = tr.ewm(alpha=1/14, adjust=False).mean()
    low14 = l.rolling(14).min()
    high14 = h.rolling(14).max()
    d["StochK"] = 100 * (c-low14) / (high14-low14).replace(0, np.nan)
    d["StochD"] = d["StochK"].rolling(3).mean()
    d["VolMA20"] = d["Volume"].rolling(20).mean()
    return d

def pivots(series, left=3, right=3, high=True):
    vals = series.to_numpy()
    out=[]
    for i in range(left, len(vals)-right):
        window=vals[i-left:i+right+1]
        if high and vals[i] == np.nanmax(window):
            out.append((i, vals[i]))
        if not high and vals[i] == np.nanmin(window):
            out.append((i, vals[i]))
    return out

def levels(df, current, kind="support"):
    col = "Low" if kind == "support" else "High"
    p = pivots(df[col].tail(180), high=(kind=="resistance"))
    vals = [float(v) for _,v in p]
    # cluster nearby pivot prices to reduce duplicate levels
    vals = sorted(vals)
    clustered=[]
    tol=max(current*0.0015, 0.5)
    for v in vals:
        if not clustered or abs(v-np.mean(clustered[-1])) > tol:
            clustered.append([v])
        else:
            clustered[-1].append(v)
    lv=[float(np.mean(x)) for x in clustered]
    if kind=="support":
        lv=[x for x in lv if x < current]
        return sorted(lv, reverse=True)[:8]
    lv=[x for x in lv if x > current]
    return sorted(lv)[:8]

def candle_patterns(df):
    if len(df)<3: return []
    a,b=df.iloc[-2],df.iloc[-1]
    body=abs(b.Close-b.Open)
    rng=max(b.High-b.Low, 1e-9)
    lower=min(b.Open,b.Close)-b.Low
    upper=b.High-max(b.Open,b.Close)
    prev_body=abs(a.Close-a.Open)
    found=[]
    # engulfing
    if b.Close>b.Open and a.Close<a.Open and b.Open<=a.Close and b.Close>=a.Open:
        found.append(("Bullish Engulfing","LONG",2))
    if b.Close<b.Open and a.Close>a.Open and b.Open>=a.Close and b.Close<=a.Open:
        found.append(("Bearish Engulfing","SHORT",2))
    if lower > body*2 and upper < body*1.2 and b.Close>b.Open:
        found.append(("Bullish Pin Bar","LONG",1))
    if upper > body*2 and lower < body*1.2 and b.Close<b.Open:
        found.append(("Bearish Pin Bar","SHORT",1))
    # 3-candle stars
    if len(df)>=3:
        x,y,z=df.iloc[-3],df.iloc[-2],df.iloc[-1]
        if x.Close<x.Open and abs(y.Close-y.Open)<abs(x.Close-x.Open)*0.45 and z.Close>z.Open and z.Close>(x.Open+x.Close)/2:
            found.append(("Morning Star","LONG",2))
        if x.Close>x.Open and abs(y.Close-y.Open)<abs(x.Close-x.Open)*0.45 and z.Close<z.Open and z.Close<(x.Open+x.Close)/2:
            found.append(("Evening Star","SHORT",2))
    return found

def structure_patterns(df):
    # Conservative heuristic flags, not a machine-learning pattern classifier.
    d=df.tail(100)
    highs=pivots(d["High"], high=True)
    lows=pivots(d["Low"], high=False)
    found=[]
    if len(lows)>=2:
        p1,p2=lows[-2][1],lows[-1][1]
        if abs(p2-p1)/max(p1,p2) < 0.004 and d.Close.iloc[-1] > max(p1,p2)*1.002:
            found.append(("Possible Double Bottom","LONG",2))
    if len(highs)>=2:
        p1,p2=highs[-2][1],highs[-1][1]
        if abs(p2-p1)/max(p1,p2) < 0.004 and d.Close.iloc[-1] < min(p1,p2)*0.998:
            found.append(("Possible Double Top","SHORT",2))
    # Wedge/flag proxies based on recent linear slopes
    if len(d)>=30:
        x=np.arange(20)
        hs=d.High.tail(20).to_numpy()
        ls=d.Low.tail(20).to_numpy()
        sh=np.polyfit(x,hs,1)[0]
        sl=np.polyfit(x,ls,1)[0]
        scale=max(float(d.Close.iloc[-1]),1)
        if sh<0 and sl<0 and abs(sh)>abs(sl)*1.15:
            found.append(("Possible Falling Wedge","LONG",1))
        if sh>0 and sl>0 and sl>sh*1.15:
            found.append(("Possible Rising Wedge","SHORT",1))
    return found

def trend_score(row):
    score=0
    if row.Close>row.EMA20: score+=1
    if row.EMA20>row.EMA50: score+=1
    if row.EMA50>row.EMA200: score+=1
    if row.MACD>row.MACDSignal: score+=1
    if row.RSI>50: score+=1
    return score

def analyze(d5,d15,d1h,d4h):
    d5,d15,d1h,d4h=[add_indicators(x).dropna() for x in (d5,d15,d1h,d4h)]
    if min(map(len,[d5,d15,d1h,d4h]))<5:
        raise ValueError("داده کافی برای تحلیل وجود ندارد.")
    r5,r15,r1,r4=[x.iloc[-1] for x in (d5,d15,d1h,d4h)]
    price=float(r15.Close)
    long_score=0; short_score=0; reasonsL=[]; reasonsS=[]
    # Higher timeframe bias
    for name,row in [("4H",r4),("1H",r1)]:
        ts=trend_score(row)
        if ts>=4:
            long_score+=2; reasonsL.append(f"روند {name} صعودی")
        elif ts<=1:
            short_score+=2; reasonsS.append(f"روند {name} نزولی")
        else:
            if row.Close>row.EMA50: long_score+=1
            else: short_score+=1
    # 15m setup
    if r15.Close>r15.EMA20>r15.EMA50:
        long_score+=2; reasonsL.append("چیدمان EMA در 15M صعودی")
    if r15.Close<r15.EMA20<r15.EMA50:
        short_score+=2; reasonsS.append("چیدمان EMA در 15M نزولی")
    if r15.MACD>r15.MACDSignal:
        long_score+=1; reasonsL.append("مومنتوم MACD مثبت")
    else:
        short_score+=1; reasonsS.append("مومنتوم MACD منفی")
    if 50<r15.RSI<70:
        long_score+=1; reasonsL.append("RSI در محدوده حمایتی صعودی")
    elif 30<r15.RSI<50:
        short_score+=1; reasonsS.append("RSI زیر 50")
    # 5m trigger
    if r5.Close>r5.EMA20 and r5.MACD>r5.MACDSignal:
        long_score+=1; reasonsL.append("تأیید ورود در 5M")
    if r5.Close<r5.EMA20 and r5.MACD<r5.MACDSignal:
        short_score+=1; reasonsS.append("تأیید ورود در 5M")
    # Volume only if feed supplies meaningful volume
    if r15.Volume>0 and pd.notna(r15.VolMA20) and r15.Volume>r15.VolMA20*1.2:
        if r15.Close>r15.Open: long_score+=1; reasonsL.append("افزایش حجم همراه کندل صعودی")
        elif r15.Close<r15.Open: short_score+=1; reasonsS.append("افزایش حجم همراه کندل نزولی")
    patterns=candle_patterns(d15)+structure_patterns(d15)
    for p,side,pts in patterns:
        if side=="LONG": long_score+=pts; reasonsL.append(f"الگو: {p}")
        else: short_score+=pts; reasonsS.append(f"الگو: {p}")
    supports=levels(d1h,price,"support")
    resistances=levels(d1h,price,"resistance")
    atr=float(r15.ATR)
    # Use nearby structural levels, fallback to ATR multiples
    def make_targets(side):
        if side=="LONG":
            sl_candidates=[x for x in supports if x<price]
            sl=max(sl_candidates) if sl_candidates else price-1.5*atr
            sl=min(sl,price-0.8*atr)
            tps=[x for x in resistances if x>price]
            fallback=[price+1.0*atr,price+1.8*atr,price+2.7*atr,price+3.8*atr]
            tps=sorted(set(tps))
            for v in fallback:
                if len(tps)>=4: break
                if not tps or v>tps[-1]+0.15*atr: tps.append(v)
            return sl,tps[:4]
        sl_candidates=[x for x in resistances if x>price]
        sl=min(sl_candidates) if sl_candidates else price+1.5*atr
        sl=max(sl,price+0.8*atr)
        tps=[x for x in supports if x<price]
        fallback=[price-1.0*atr,price-1.8*atr,price-2.7*atr,price-3.8*atr]
        tps=sorted(set(tps),reverse=True)
        for v in fallback:
            if len(tps)>=4: break
            if not tps or v<tps[-1]-0.15*atr: tps.append(v)
        return sl,tps[:4]
    direction="WAIT"
    if long_score>=7 and long_score-short_score>=2: direction="LONG"
    elif short_score>=7 and short_score-long_score>=2: direction="SHORT"
    side=direction if direction!="WAIT" else ("LONG" if long_score>short_score else "SHORT")
    sl,tps=make_targets(side)
    risk=abs(price-sl)
    targets=[]
    for i,tp in enumerate(tps,1):
        reward=abs(tp-price)
        targets.append({"هدف":f"TP{i}","قیمت":tp,"سود/زیان قیمت (%)":reward/price*100,
                        "R/R":reward/risk if risk else np.nan})
    score=max(long_score,short_score)
    confidence=min(95,round(score/12*100))
    return {"direction":direction,"long_score":long_score,"short_score":short_score,
            "confidence":confidence,"price":price,"sl":sl,"targets":targets,
            "reasons_long":reasonsL,"reasons_short":reasonsS,"patterns":patterns,
            "supports":supports,"resistances":resistances,"atr":atr,
            "rsi":float(r15.RSI),"macd":float(r15.MACD),"macds":float(r15.MACDSignal),
            "d15":d15}

# ----------------------- UI -----------------------
st.title("🥇 Gold AI Trading Scanner")
st.caption("تحلیل چندتایم‌فریمی XAUUSD | سیگنال LONG و SHORT | نسخه آزمایشی")
with st.sidebar:
    st.header("تنظیمات")
    ticker=st.selectbox("نماد داده",["XAUUSD=X","GC=F"],index=0,
        help="XAUUSD=X قیمت نقدی طلا در یاهو فایننس است. GC=F قرارداد آتی طلا و صرفاً جایگزین است.")
    min_score=st.slider("حداقل امتیاز سیگنال",5,10,7)
    account=st.number_input("سرمایه حساب (USD)",min_value=10.0,value=1000.0,step=100.0)
    risk_pct=st.slider("ریسک هر معامله (%)",0.25,2.0,1.0,0.25)
    leverage_cap=st.slider("حداکثر اهرم مجاز",1,50,10)
    refresh=st.button("🔄 دریافت و تحلیل مجدد")

if refresh:
    st.cache_data.clear()

with st.spinner("در حال دریافت داده‌های بازار و محاسبه اندیکاتورها..."):
    d5=get_data(ticker,"5m","5d")
    d15=get_data(ticker,"15m","60d")
    d1h=get_data(ticker,"1h","730d")
    if not d1h.empty:
        d4h=d1h.resample("4h").agg({"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}).dropna()
    else:
        d4h=pd.DataFrame()

if any(x.empty for x in [d5,d15,d1h,d4h]):
    st.error("دریافت داده از Yahoo Finance ناموفق بود یا داده کافی نیست. چند دقیقه بعد دوباره تلاش کنید؛ ممکن است دسترسی Yahoo در بعضی شبکه‌ها محدود باشد.")
    st.info("این نسخه از Yahoo Finance استفاده می‌کند و قیمت/داده ممکن است با بروکر شما اختلاف جزئی داشته باشد.")
    st.stop()

try:
    result=analyze(d5,d15,d1h,d4h)
except Exception as e:
    st.error(f"خطا در تحلیل: {e}")
    st.stop()

# Apply user threshold after analysis
direction=result["direction"]
if direction=="LONG" and (result["long_score"]<min_score or result["long_score"]-result["short_score"]<2):
    direction="WAIT"
if direction=="SHORT" and (result["short_score"]<min_score or result["short_score"]-result["long_score"]<2):
    direction="WAIT"

c1,c2,c3,c4=st.columns(4)
c1.metric("قیمت فعلی",f'{result["price"]:,.2f}')
c2.metric("وضعیت بازار",direction,delta=f'LONG {result["long_score"]} | SHORT {result["short_score"]}')
c3.metric("اعتبار نسبی سیگنال",f'{result["confidence"]}%')
c4.metric("RSI (15M)",f'{result["rsi"]:.1f}')

st.subheader("📍 پیشنهاد معاملاتی")
if direction=="WAIT":
    st.warning("فعلاً ورود نکنید؛ امتیاز یا اختلاف امتیاز دو جهت برای صدور سیگنال کافی نیست.")
else:
    is_long=direction=="LONG"
    sl=result["sl"]
    risk=abs(result["price"]-sl)
    risk_cash=account*risk_pct/100
    units=risk_cash/risk if risk>0 else 0
    notional=units*result["price"]
    leverage=max(1,min(leverage_cap,int(np.ceil(notional/account)))) if account else 1
    a,b,c,d=st.columns(4)
    a.metric("جهت",direction)
    b.metric("Entry",f'{result["price"]:,.2f}')
    c.metric("Stop Loss",f'{sl:,.2f}',delta=f'{(sl/result["price"]-1)*100:+.2f}%')
    d.metric("حجم تقریبی (اونس)",f"{units:.3f}")
    tp_df=pd.DataFrame(result["targets"])
    if not tp_df.empty:
        tp_df["سود/زیان قیمت (%)"]=tp_df["سود/زیان قیمت (%)"].map(lambda x: f"{x:.2f}%")
        tp_df["قیمت"]=tp_df["قیمت"].map(lambda x:f"{x:,.2f}")
        tp_df["R/R"]=tp_df["R/R"].map(lambda x:f"{x:.2f}")
        st.dataframe(tp_df,use_container_width=True,hide_index=True)
    st.caption(f"ریسک دلاری هدف‌گذاری‌شده: {risk_cash:.2f} USD | ارزش اسمی تقریبی: {notional:,.0f} USD | اهرم محاسباتی تقریبی: {leverage}x")
    st.caption("اهرم صرفاً برآورد ریاضی است؛ مارجین، کارمزد، اسلیپیج، قرارداد بروکر و فاصله لیکوییدیشن در این نسخه لحاظ نشده‌اند.")

left,right=st.columns(2)
with left:
    st.markdown("#### 🟢 دلایل LONG")
    for x in result["reasons_long"]: st.write("•",x)
    if not result["reasons_long"]: st.write("دلیل تأییدشده‌ای ثبت نشد.")
with right:
    st.markdown("#### 🔴 دلایل SHORT")
    for x in result["reasons_short"]: st.write("•",x)
    if not result["reasons_short"]: st.write("دلیل تأییدشده‌ای ثبت نشد.")

st.subheader("الگوهای شناسایی‌شده")
if result["patterns"]:
    st.dataframe(pd.DataFrame(result["patterns"],columns=["الگو","جهت","وزن امتیاز"]),use_container_width=True,hide_index=True)
else:
    st.write("در آخرین کندل‌های بسته‌شده الگوی مشخصی شناسایی نشد.")

st.subheader("حمایت و مقاومت")
s1,s2=st.columns(2)
with s1: st.write("**حمایت‌ها**"); st.write([round(x,2) for x in result["supports"]] or "یافت نشد")
with s2: st.write("**مقاومت‌ها**"); st.write([round(x,2) for x in result["resistances"]] or "یافت نشد")

chart=result["d15"].tail(150)
fig=go.Figure(data=[go.Candlestick(x=chart.index,open=chart.Open,high=chart.High,low=chart.Low,close=chart.Close,name="XAUUSD")])
fig.add_trace(go.Scatter(x=chart.index,y=chart.EMA20,name="EMA20",line=dict(width=1)))
fig.add_trace(go.Scatter(x=chart.index,y=chart.EMA50,name="EMA50",line=dict(width=1)))
if direction in ("LONG","SHORT"):
    fig.add_hline(y=result["price"],line_dash="dot",annotation_text="Entry")
    fig.add_hline(y=result["sl"],line_dash="dash",annotation_text="SL")
    for t in result["targets"]:
        fig.add_hline(y=t["قیمت"],line_dash="dot",annotation_text=t["هدف"])
fig.update_layout(height=600,xaxis_rangeslider_visible=False,template="plotly_dark",margin=dict(l=10,r=10,t=30,b=10))
st.plotly_chart(fig,use_container_width=True)

st.subheader("جزئیات شاخص‌ها")
st.write(f'ATR (15M): {result["atr"]:.3f} | MACD: {result["macd"]:.4f} | Signal: {result["macds"]:.4f}')
st.caption(f"آخرین به‌روزرسانی: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | منبع: Yahoo Finance")

with st.expander("روش‌شناسی و محدودیت‌ها"):
    st.markdown("""
    - امتیازدهی یک سیستم قاعده‌محور و آزمایشی است؛ «احتمال آماری موفقیت» یا تضمین سود نیست.
    - تایم‌فریم‌های 5M، 15M و 1H از داده‌های Yahoo Finance گرفته می‌شوند؛ 4H از تجمیع کندل‌های 1H ساخته می‌شود.
    - شناسایی الگوها الگوریتمی و تقریبی است؛ تأیید بصری معامله‌گر همچنان لازم است.
    - اهداف ابتدا از پیوت‌های حمایت/مقاومت استخراج می‌شوند و در صورت کمبود سطوح، ATR نقش جایگزین دارد.
    - حجم XAUUSD در بسیاری از فیدهای فارکس حجم واقعی بازار متمرکز نیست؛ ممکن است صفر یا حجم تیک باشد.
    - پیش از استفاده واقعی، بک‌تست، حساب دمو، لحاظ اسپرد/کمیسیون/اسلیپیج و تطبیق قیمت با بروکر ضروری است.
    """)
