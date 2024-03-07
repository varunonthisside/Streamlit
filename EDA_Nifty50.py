import streamlit as st
import pandas as pd
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import yfinance as yf

st.title('Nifty 50 App')

st.markdown("""
This app retrieves the list of the **Nifty 50 stocks** (from Wikipedia) 
* **Python libraries:** base64, pandas, streamlit, numpy, matplotlib, seaborn
* **Data source:** [Wikipedia](https://en.wikipedia.org/wiki/NIFTY_50).
""")

st.sidebar.header('User Input Features')

# Web scraping of Nifty 50 data
#
@st.cache_data
def load_data():
    url = 'https://en.wikipedia.org/wiki/NIFTY_50'
    html = pd.read_html(url, header = 0)
    df = html[2]
    return df

df = load_data()
df.columns=['Company Name', 'Symbol', 'Sector', 'Date Added']
sector = df.groupby('Sector')

# Sidebar - Sector selection
sorted_sector_unique = sorted( df['Sector'].unique() )
selected_sector = st.sidebar.multiselect('Sector', sorted_sector_unique, sorted_sector_unique)

# Filtering data
df_selected_sector = df[ (df['Sector'].isin(selected_sector)) ]

st.header('Display Companies in Selected Sector')
st.write('Data Dimension: ' + str(df_selected_sector.shape[0]) + ' rows and ' + str(df_selected_sector.shape[1]) + ' columns.')
st.dataframe(df_selected_sector)

# Download Nifty50 data
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="nifty50.csv">Download CSV File</a>'
    return href

st.markdown(filedownload(df_selected_sector), unsafe_allow_html=True)

# https://pypi.org/project/yfinance/

tickers = list(df_selected_sector[:10].Symbol)
tickers_nse = list([symbol + '.NS' for symbol in tickers])

data = yf.download(
        tickers = tickers_nse,
        period = "ytd",
        interval = "1d",
        group_by = 'ticker',
        auto_adjust = True,
        prepost = True,
        threads = True,
        proxy = None
    )

# Plot Closing Price of Query Symbol
def price_plot(symbol):
  df = pd.DataFrame(data[symbol].Close)
  df['Date'] = df.index
  fig, ax = plt.subplots(figsize=(10,10))
  plt.fill_between(df.Date, df.Close, color='skyblue', alpha=0.3)
  plt.plot(df.Date, df.Close, color='skyblue', alpha=0.8)
  plt.xticks(rotation=90)
  plt.title(symbol, fontweight='bold')
  plt.xlabel('Date', fontweight='bold')
  plt.ylabel('Closing Price', fontweight='bold')
  return st.pyplot(fig)

num_company = st.sidebar.slider('Number of Companies', 1, 5)

if st.button('Show Plots'):
    st.header('Stock Closing Price')
    for i in tickers_nse[:num_company]:
        price_plot(i)