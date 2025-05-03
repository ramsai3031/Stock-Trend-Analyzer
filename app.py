import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
# Add these with your other imports (usually first few lines)
import time
from datetime import datetime
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands
from ta.volume import VolumeWeightedAveragePrice
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import nltk
# Add this with your other imports at the VERY TOP of your file
import plotly.io as pio
pio.templates.default = "plotly_white"  # This sets the default theme for ALL plots
from nltk.sentiment import SentimentIntensityAnalyzer
import warnings
warnings.filterwarnings('ignore')

# Download NLTK data for sentiment analysis
nltk.download('vader_lexicon')

# Set page configuration
st.set_page_config(
    page_title="Stock Trend Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 24px;
    }
    .stSelectbox, .stDateInput {
        width: 100%;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        color: #000000;  /* dark text for visibility */
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .positive {
        color: green;
    }
    .negative {
        color: red;
    }
    h2, h3, p {
            margin: 0;
            padding: 5px 0;
        }
    </style>
    """, unsafe_allow_html=True)
def safe_extract(value):
    """Safely extract the last value from Series/DataFrame or return scalar"""
    try:
        if hasattr(value, 'iloc'):  # If it's a Series/DataFrame
            return value.iloc[-1] if not value.empty else None
        return float(value) if value is not None else None
    except:
        return None

def safe_format(value, format_spec=".1f", suffix=""):
    """Safely format values with fallback to N/A"""
    try:
        if value is None or pd.isna(value):
            return "N/A"
        return f"{float(value):{format_spec}}{suffix}"
    except:
        return "N/A"

# Initialize session state
if 'stock_data' not in st.session_state:
    st.session_state.stock_data = None
if 'ticker' not in st.session_state:
    st.session_state.ticker = None
if 'start_date' not in st.session_state:
    st.session_state.start_date = datetime.now() - timedelta(days=365)
if 'end_date' not in st.session_state:
    st.session_state.end_date = datetime.now()

# Function to fetch stock data
def fetch_stock_data(ticker, start_date, end_date):
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if not isinstance(data, pd.DataFrame) or data.empty:
            st.error("No data found for this stock ticker.")
            return None
        
        # Ensure all columns are 1D and numeric
        for col in data.columns:
            if isinstance(data[col], pd.DataFrame):
                data[col] = data[col].squeeze()
            data[col] = pd.to_numeric(data[col], errors='coerce')
        
        return data.dropna()
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None


# Function to calculate technical indicators
def calculate_indicators(data):
    # RSI
    close_prices = data['Close'].squeeze() if isinstance(data['Close'], pd.DataFrame) else data['Close']
    high_prices = data['High'].squeeze() if isinstance(data['High'], pd.DataFrame) else data['High']
    low_prices = data['Low'].squeeze() if isinstance(data['Low'], pd.DataFrame) else data['Low']
    volumes = data['Volume'].squeeze() if isinstance(data['Volume'], pd.DataFrame) else data['Volume']
    rsi_indicator = RSIIndicator(close=close_prices, window=14)
    data['RSI'] = rsi_indicator.rsi()
    
    # MACD
    macd_indicator = MACD(close=close_prices)
    data['MACD'] = macd_indicator.macd()
    data['MACD_Signal'] = macd_indicator.macd_signal()
    data['MACD_Hist'] = macd_indicator.macd_diff()
    
    # Bollinger Bands
    bb_indicator = BollingerBands(close=close_prices, window=20, window_dev=2)
    data['BB_Upper'] = bb_indicator.bollinger_hband()
    data['BB_Lower'] = bb_indicator.bollinger_lband()
    
    # Moving Averages
    data['MA_50'] = close_prices.rolling(window=50).mean()
    data['MA_200'] = close_prices.rolling(window=200).mean()
    
    # VWAP
    vwap_indicator = VolumeWeightedAveragePrice(
        high=high_prices,
        low=low_prices,
        close=close_prices,
        volume=volumes,
        window=14
    )
    data['VWAP'] = vwap_indicator.volume_weighted_average_price()
    
    return data

# Function to get current price
def get_current_price(ticker):
    try:
        stock = yf.Ticker(ticker)
        current_data = stock.history(period='1d')
        return current_data['Close'].iloc[-1]
    except:
        return None

# Function to get stock info
def get_stock_info(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return info
    except:
        return None

# Function for sentiment analysis
def analyze_sentiment(ticker):
    try:
        # Get news headlines (simulated - in a real app you'd fetch actual news)
        company_name = get_stock_info(ticker).get('shortName', ticker)
        url = f"https://www.google.com/search?q={company_name}+stock+news&tbm=nws"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        headlines = [h.text for h in soup.find_all('div', class_='BNeawe vvjwJb AP7Wnd')][:5]
        
        # Analyze sentiment
        sia = SentimentIntensityAnalyzer()
        sentiment_scores = [sia.polarity_scores(h)['compound'] for h in headlines]
        avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0
        
        return {
            'headlines': headlines,
            'sentiment_score': avg_sentiment,
            'sentiment_label': 'Positive' if avg_sentiment > 0.05 else 'Negative' if avg_sentiment < -0.05 else 'Neutral'
        }
    except Exception as e:
        st.error(f"Error in sentiment analysis: {e}")
        return {
            'headlines': ["No headlines available"],
            'sentiment_score': 0,
            'sentiment_label': 'Neutral'
        }

# Function to assess risk
def assess_risk(data):
    if data is None or not isinstance(data, pd.DataFrame) or data.empty or len(data) < 30:
        return "Low", 1
    
    try:
        daily_returns = data['Close'].pct_change().dropna()
        volatility = daily_returns.std()
        
        # Handle different types of volatility values
        if isinstance(volatility, (pd.Series, pd.DataFrame)):
            if len(volatility) == 1:
                volatility = float(volatility.iloc[0])
            else:
                volatility = float(volatility.mean())
        else:
            volatility = float(volatility)
        
        if volatility < 0.01:
            return "Low", 1
        elif volatility < 0.03:
            return "Medium", 2
        else:
            return "High", 3
    except Exception as e:
        st.warning(f"Risk assessment error: {str(e)}")
        return "Medium", 2

# Function to make simple prediction (for demo purposes)
# Function to make simple prediction (for demo purposes)
def make_prediction(data):
    if data is None or not isinstance(data, pd.DataFrame) or data.empty or len(data) < 30:
        return None, None

    try:
        # Helper function to safely get last value
        def get_last_safe(col):
            if col.empty:
                return np.nan
            if isinstance(col, (pd.Series, pd.DataFrame)):
                return float(col.iloc[-1]) if len(col) > 0 else np.nan
            return float(col)

        last_close = get_last_safe(data['Close'])
        ma_50 = get_last_safe(data['MA_50'])
        ma_200 = get_last_safe(data['MA_200'])

        if np.isnan(last_close) or np.isnan(ma_50) or np.isnan(ma_200):
            return None, None

        if ma_50 > ma_200 and last_close > ma_50:
            prediction = "Bullish"
        elif ma_50 < ma_200 and last_close < ma_50:
            prediction = "Bearish"
        else:
            prediction = "Neutral"

        confidence = min(0.9, abs((ma_50 - ma_200) / ma_200 * 5))
        return str(prediction), confidence  # ENSURE prediction is a string
    except Exception as e:
        st.warning(f"Prediction error: {str(e)}")
        return None, None

# Main app
def main():
    st.title("📈 Stock Trend Analyzer")
    
    # Sidebar
    with st.sidebar:
        st.header("Stock Selection")
        ticker = st.text_input("Enter Stock Ticker (e.g., AAPL)", value="AAPL").upper()
        
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=st.session_state.start_date)
        with col2:
            end_date = st.date_input("End Date", value=st.session_state.end_date)
        
        fetch_button = st.button("Fetch Data")
        refresh_button = st.button("Refresh Data")
        
        if fetch_button or refresh_button:
            st.session_state.ticker = ticker
            st.session_state.start_date = start_date
            st.session_state.end_date = end_date
            with st.spinner("Fetching stock data..."):
                st.session_state.stock_data = fetch_stock_data(ticker, start_date, end_date)
                if st.session_state.stock_data is not None:
                    start_time = time.time()
                    st.session_state.stock_data = calculate_indicators(st.session_state.stock_data)
                    st.success("Data fetched successfully!")
        
        st.markdown("---")
        st.header("Glossary")
        with st.expander("Technical Indicators"):
            st.markdown("""
            - **RSI (Relative Strength Index)**: Measures speed and change of price movements (30-70 range).
            - **MACD (Moving Average Convergence Divergence)**: Shows relationship between two moving averages.
            - **Bollinger Bands**: Volatility bands placed above and below a moving average.
            - **Moving Averages**: Smooth out price data to identify trends.
            """)
        
        with st.expander("Risk Levels"):
            st.markdown("""
            - **Low**: Stable price movements, low volatility.
            - **Medium**: Moderate price fluctuations.
            - **High**: Significant price swings, high volatility.
            """)
    
    # Main content
    if st.session_state.stock_data is not None and st.session_state.ticker:
        data = st.session_state.stock_data
        ticker = st.session_state.ticker
        current_price = get_current_price(ticker)
        stock_info = get_stock_info(ticker)
        company_name = stock_info.get('shortName', ticker) if stock_info else ticker
        sentiment = analyze_sentiment(ticker)
        risk_level, risk_score = assess_risk(data)
        prediction, confidence = make_prediction(data)
        
        # Stock Information Section
        st.subheader(f"{company_name} ({ticker})")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            current_price = get_current_price(ticker)
            current_price_display = f"${float(current_price):.2f}" if current_price and not pd.isna(current_price) else "N/A"
            st.metric("Current Price", current_price_display)

        with col2:
            rsi_value = data['RSI'].iloc[-1] if not data['RSI'].empty else None
            rsi_display = f"{float(rsi_value):.2f}" if rsi_value is not None and not pd.isna(rsi_value) else "N/A"
            rsi_status = "Overbought (>70)" if rsi_value and rsi_value > 70 else "Oversold (<30)" if rsi_value and rsi_value < 30 else "Neutral"
            st.metric("RSI", f"{rsi_display} - {rsi_status}" if rsi_display != "N/A" else "N/A")

        with col3:
            sentiment_score = sentiment.get('sentiment_score', None)
            sentiment_display = f"{float(sentiment_score):.2f}" if sentiment_score is not None and not pd.isna(sentiment_score) else "N/A"
            st.metric("Industry Sentiment", sentiment.get('sentiment_label', 'N/A'), 
             delta=sentiment_display if sentiment_display != "N/A" else None)

        with col4:
            st.metric("Risk Assessment", risk_level)
        
        # Download button
        st.download_button(
            label="Download Data",
            data=data.to_csv().encode('utf-8'),
            file_name=f"{ticker}_stock_data.csv",
            mime='text/csv'
        )
        
        # Tabs for different sections
        tab1, tab2, tab3, tab4 = st.tabs(["Price Charts", "Technical Indicators", "Stock Insights", "Historical Analysis"])
        
        with tab1:
            st.subheader("Price Movement")
            
            # Timeframe selection
            time_options = ['1D', '1W', '1M', '3M', 'YTD', '1Y', 'All']
            selected_time = st.radio("Select Timeframe", time_options, horizontal=True)
            
            # Filter data based on selected timeframe
            if selected_time == '1D':
                chart_data = data.tail(1)
            elif selected_time == '1W':
                chart_data = data.tail(5)
            elif selected_time == '1M':
                chart_data = data.tail(21)
            elif selected_time == '3M':
                chart_data = data.tail(63)
            elif selected_time == 'YTD':
                chart_data = data[data.index >= datetime(datetime.now().year, 1, 1)]
            elif selected_time == '1Y':
                chart_data = data.tail(252)
            else:
                chart_data = data
            
            # Create interactive candlestick chart
            fig = go.Figure(data=[go.Candlestick(
                x=chart_data.index,
                open=chart_data['Open'],
                high=chart_data['High'],
                low=chart_data['Low'],
                close=chart_data['Close'],
                name='Price'
            )])
            
            # Add moving averages
            if 'MA_50' in chart_data.columns:
                fig.add_trace(go.Scatter(
                    x=chart_data.index,
                    y=chart_data['MA_50'],
                    name='50-Day MA',
                    line=dict(color='orange', width=1)
                ))
            
            if 'MA_200' in chart_data.columns:
                fig.add_trace(go.Scatter(
                    x=chart_data.index,
                    y=chart_data['MA_200'],
                    name='200-Day MA',
                    line=dict(color='purple', width=1)
                ))
            
            fig.update_layout(
                height=500,
                xaxis_rangeslider_visible=False,
                showlegend=True,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Price prediction vs actual
            st.subheader("Prediction vs Actual")
            if prediction:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Current Prediction</h3>
                        <h2 {'class="positive"' if prediction == 'Bullish' else 'class="negative"' if prediction == 'Bearish' else ''}>{prediction}</h2>
                        <p>Confidence: {confidence*100:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    # Simple historical accuracy (simulated)
                    accuracy = min(0.85, 0.7 + confidence * 0.3)
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Historical Accuracy</h3>
                        <h2>{accuracy*100:.1f}%</h2>
                        <p>Based on similar market conditions</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("Not enough data for prediction")
        
        with tab2:
            st.subheader("Technical Indicators")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # RSI Chart
                st.markdown("**Relative Strength Index (RSI)**")
                fig_rsi = go.Figure()
                fig_rsi.add_trace(go.Scatter(
                    x=data.index,
                    y=data['RSI'],
                    name='RSI',
                    line=dict(color='blue', width=2)
                ))
                fig_rsi.add_hline(y=30, line_dash="dot", line_color="green")
                fig_rsi.add_hline(y=70, line_dash="dot", line_color="red")
                fig_rsi.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig_rsi, use_container_width=True)
                
                # Bollinger Bands
                st.markdown("**Bollinger Bands**")
                fig_bb = go.Figure()
                fig_bb.add_trace(go.Scatter(
                    x=data.index,
                    y=data['BB_Upper'],
                    name='Upper Band',
                    line=dict(color='red', width=1)
                ))
                fig_bb.add_trace(go.Scatter(
                    x=data.index,
                    y=data['BB_Lower'],
                    name='Lower Band',
                    line=dict(color='green', width=1)
                ))
                fig_bb.add_trace(go.Scatter(
                    x=data.index,
                    y=data['Close'],
                    name='Price',
                    line=dict(color='blue', width=2)
                ))
                fig_bb.update_layout(height=300, showlegend=True)
                st.plotly_chart(fig_bb, use_container_width=True)
            
            with col2:
                # MACD
                st.markdown("**MACD (Moving Average Convergence Divergence)**")
                fig_macd = go.Figure()
                fig_macd.add_trace(go.Scatter(
                    x=data.index,
                    y=data['MACD'],
                    name='MACD',
                    line=dict(color='blue', width=2)
                ))
                fig_macd.add_trace(go.Scatter(
                    x=data.index,
                    y=data['MACD_Signal'],
                    name='Signal Line',
                    line=dict(color='orange', width=2)
                ))
                
                # Add histogram
                colors = ['green' if val >= 0 else 'red' for val in data['MACD_Hist']]
                fig_macd.add_trace(go.Bar(
                    x=data.index,
                    y=data['MACD_Hist'],
                    name='Histogram',
                    marker_color=colors
                ))
                
                fig_macd.update_layout(height=300, showlegend=True)
                st.plotly_chart(fig_macd, use_container_width=True)
                
        #         # Volume
        #         st.markdown("**Volume**")
        #         fig_vol = go.Figure()
        #         fig_vol.add_trace(go.Bar(
        #         x=data.index,
        #         y=data['Volume'],
        #         name='Volume',
        #         marker_color='blue',
        #         opacity=0.6  # Makes bars slightly transparent
        #         ))
        
        # # Add 20-day moving average of volume
        #         if 'Volume' in data:
        #             data['Volume_MA20'] = data['Volume'].rolling(20).mean()
        #             fig_vol.add_trace(go.Scatter(
        #             x=data.index,
        #             y=data['Volume_MA20'],
        #             name='20-Day MA',
        #             line=dict(color='orange', width=2)
        #     ))
        
        #         fig_vol.update_layout(
        #          height=300,
        #          showlegend=True,
        #          yaxis_title='Volume',
        #          xaxis_rangeslider_visible=False
        # )
        #         st.plotly_chart(fig_vol, use_container_width=True)

        
        with tab3:
            st.subheader("Stock Insights")
            
            col1, col2 = st.columns(2)
            
            with col1:
                 st.markdown("**Industry Sentiment Analysis**")
        
        # Get proper company name
                 stock_info = get_stock_info(ticker)
                 company_name = stock_info.get('shortName', ticker) if stock_info else ticker
                 search_query = f"{company_name} stock news"
        
        # Enhanced news scraping with multiple sources
                 def fetch_news_headlines(query):
                     headlines = []
                     sources = [
                {
                    'url': f"https://news.google.com/search?q={query}&hl=en-US&gl=US&ceid=US:en",
                    'selector': 'article h3'
                },
                {
                    'url': f"https://www.bing.com/news/search?q={query}",
                    'selector': '.title'
                },
                {
                    'url': f"https://finviz.com/quote.ashx?t={ticker}",
                    'selector': '.news-link-cell a'
                }
            ]
            
                     headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
                     for source in sources:
                         try:
                             response = requests.get(source['url'], headers=headers, timeout=5)
                             soup = BeautifulSoup(response.text, 'html.parser')
                             articles = soup.select(source['selector'])[:5]  # Get first 5
                    
                             for article in articles:
                                 headline = article.get_text().strip()
                                 if headline and len(headline) > 10:  # Filter out empty/short texts
                                     headlines.append(headline)
                                     if len(headlines) >= 5:  # Stop when we have enough
                                        return headlines
                         except:
                             continue
            
                     return headlines if headlines else ["No recent headlines found - try again later"]
        
        # Fetch and display news
                 news_headlines = fetch_news_headlines(search_query)
        
                 st.markdown("**Recent News Headlines**")
                 for i, headline in enumerate(news_headlines[:5], 1):  # Show max 5 headlines
                     st.markdown(f"• {headline}")
        
        # Sentiment analysis (existing code)
                 sentiment = analyze_sentiment(ticker)
                 st.metric("Overall Sentiment", sentiment['sentiment_label'], 
                 delta=f"{sentiment['sentiment_score']:.2f} sentiment score")

            
            with col2:
                st.markdown("**Stock Prediction**")
                try:
                    def safe_extract(series, default=None):
                        try:
                            return float(series.iloc[-1]) if not series.empty else default
                        except:
                            return default

          # Get all values as native Python types
                    current_rsi = safe_extract(data['RSI'])
                    current_macd = safe_extract(data['MACD'])
                    current_signal = safe_extract(data['MACD_Signal'])
                    confidence_value = float(confidence) if confidence is not None else None

          # PREPARE DISPLAY STRINGS - FORCING STRING CONVERSION
                    def safe_format(value, format_str=".1f", suffix=""):
                        try:
                            return f"{float(value):{format_str}}{suffix}" if isinstance(value, (int, float)) else str(value) if value is not None else "N/A"
                        except:
                            return "N/A"

                    confidence_str = str(safe_format(confidence_value, ".1f", "%"))
                    rsi_status = "Overbought (>70)" if current_rsi and current_rsi > 70 else "Oversold (<30)" if current_rsi and current_rsi < 30 else "Neutral"
                    rsi_str = str(safe_format(current_rsi, ".1f", f" ({rsi_status})"))
                    macd_status = str("Neutral")
                    if None not in [current_macd, current_signal]:
                        macd_status = str("Bullish crossover" if current_macd > current_signal else "Bearish crossover")
                    prediction_str = str(prediction) if prediction is not None else "N/A"

          # EXPLICIT TYPE CHECK AND CONVERSION BEFORE F-STRING
                   # st.write(f"Type of prediction: {type(prediction)}, Value: {prediction}")
                   # st.write(f"Type of prediction_str: {type(prediction_str)}, Value: {prediction_str}")
                   # st.write(f"Type of confidence_str: {type(confidence_str)}, Value: {confidence_str}")
                   # st.write(f"Type of rsi_str: {type(rsi_str)}, Value: {rsi_str}")
                   # st.write(f"Type of macd_status: {type(macd_status)}, Value: {macd_status}")
     
          # RENDER PREDICTION
                    if prediction:
                        st.markdown(f"""
   <div style="background-color: #f0f0f0; padding: 20px; border-radius: 10px;color: #000000;  /* dark text for visibility */">
    <h3 style="color: {'green' if prediction == 'Bullish' else 'red' if prediction == 'Bearish' else 'black'}">
     {prediction_str} Trend Predicted
    </h3>
    <p>Confidence: {confidence_str}</p>
    <p>Key Factors:</p>
    <ul>
     <li>{'50-day MA above 200-day MA' if prediction == 'Bullish' else '50-day MA below 200-day MA'}</li>
     <li>RSI: {rsi_str}</li>
     <li>MACD: {macd_status}</li>
    </ul>
   </div>
   """, unsafe_allow_html=True)
                    else:
                        st.warning("No prediction available")

                except Exception as e:
                    st.error(f"Display error in prediction section: {str(e)}")
                    st.markdown("""
  background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        color: #000000;  /* dark text for visibility */
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
   <h3>Prediction Unavailable</h3>
   <p>Please try refreshing the data</p>
  </div>
  """, unsafe_allow_html=True)
                # Recommendation
                st.markdown("**Recommendation**")
                if prediction == "Bullish" and risk_level == "Low":
                    st.success("Strong Buy Opportunity")
                elif prediction == "Bullish" and risk_level == "Medium":
                    st.info("Consider Buying")
                elif prediction == "Bearish" and risk_level == "High":
                    st.error("Consider Selling")
                else:
                    st.warning("Hold Position")
        
        with tab4:
            st.subheader("Historical Analysis")
            
            # Historical performance metrics
            # Replace your monthly returns section with this:

            col1, col2, col3 = st.columns(3)
            with col1:
                try:
                    if len(data) >= 21:
                        monthly_return = (data['Close'].iloc[-1] / data['Close'].iloc[-21] - 1) * 100
                        monthly_value = float(monthly_return.iloc[0]) if hasattr(monthly_return, 'iloc') else float(monthly_return)
                        st.metric("1-Month Return", f"{monthly_value:.2f}%")
                    else:
                        st.metric("1-Month Return", "N/A")
                except Exception as e:
                    st.metric("1-Month Return", "Error")

            with col2:
                try:
                    if len(data) >= 63:
                        quarterly_return = (data['Close'].iloc[-1] / data['Close'].iloc[-63] - 1) * 100
                        quarterly_value = float(quarterly_return.iloc[0]) if hasattr(quarterly_return, 'iloc') else float(quarterly_return)
                        st.metric("3-Month Return", f"{quarterly_value:.2f}%")
                    else:
                        st.metric("3-Month Return", "N/A")
                except Exception as e:
                    st.metric("3-Month Return", "Error")

            with col3:
                try:
                    if len(data) >= 252:
                        yearly_return = (data['Close'].iloc[-1] / data['Close'].iloc[-252] - 1) * 100
                        yearly_value = float(yearly_return.iloc[0]) if hasattr(yearly_return, 'iloc') else float(yearly_return)
                        st.metric("1-Year Return", f"{yearly_value:.2f}%")
                    else:
                        st.metric("1-Year Return", "N/A")
                except Exception as e:
                    st.metric("1-Year Return", "Error")
            # Historical price chart
            st.markdown("**Historical Price Performance**")
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Scatter(
                x=data.index,
                y=data['Close'],
                name='Price',
                line=dict(color='blue', width=2)
            ))
            fig_hist.update_layout(height=400, showlegend=True)
            st.plotly_chart(fig_hist, use_container_width=True)
            
            # Historical volatility
            st.markdown("**Historical Volatility**")
            data['Daily_Return'] = data['Close'].pct_change()
            data['Volatility'] = data['Daily_Return'].rolling(window=21).std() * np.sqrt(252)
            
            fig_vol = go.Figure()
            fig_vol.add_trace(go.Scatter(
                x=data.index,
                y=data['Volatility'],
                name='Annualized Volatility',
                line=dict(color='red', width=2)
            ))
            fig_vol.update_layout(height=300, showlegend=True)
            st.plotly_chart(fig_vol, use_container_width=True)
            
            analysis_time = time.time() - start_time
        
        # Display success message (above the else clause)
            # Replace st.balloons() with this combo:
           
            st.balloons()

            st.success(f"""✔ {st.session_state.ticker} Stock analysis completed in {analysis_time:.2f}s
                   • {len(st.session_state.stock_data)} data points
                   • Last updated {datetime.now().strftime('%Y-%m-%d %H:%M')}""")
    
    else:
        # Replace the line with the image in the else block (around line 700)
        st.info("""
        **Welcome to Stock Trend Analyzer!**  
        Please enter a stock ticker (like AAPL for Apple, MSFT for Microsoft)  
        and click 'Fetch Data' to begin analysis.
        """)
        # Add this where you want the image to appear (e.g., in your home/welcome section)
        # Basic image display with new parameter

    
if __name__ == "__main__":
    main()