import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from openai import OpenAI
import plotly.graph_objects as go

# Preset for the OpenAI model
preset = """You now act as an agent to help me transform user's input to a formatted dictionary. 
The user should only request for financial stock market data. If it is not the case, you return a message 'ERROR'. 
If the user indeed asks for financial data, for example, give me the stock of apple in recent 60 days. 
Then you return a dictionary {'stock': 'AAPL', 'time': 60}.
If no date information is found, then you set it by default to 60"""

def generate_response(api_key, prompt, model="gpt-3.5-turbo", max_tokens=100):
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": preset},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens
    )
    return response.choices[0].message.content

def plot_stock_gpt(api_key, prompt):
    response = generate_response(api_key, prompt)
    try:
        input_dict = eval(response)
        if not isinstance(input_dict, dict):
            return "Invalid input! Try again!"
    except:
        return "Invalid input!"
    
    return plot_stock(input_dict['stock'], input_dict['time'])

def plot_stock(stock_name, time_period):
    end_date = pd.to_datetime('today')
    start_date = end_date - pd.Timedelta(days=time_period)
    stock = yf.Ticker(stock_name)
    stock_data = stock.history(start=start_date, end=end_date)

    fig = go.Figure(data=[go.Candlestick(x=stock_data.index,
                open=stock_data['Open'],
                high=stock_data['High'],
                low=stock_data['Low'],
                close=stock_data['Close'])])

    fig.update_layout(
        title=f'{stock_name} Stock Prices for the Last {time_period} Days',
        yaxis_title='Price',
        xaxis_title='Date',
        template='plotly_dark'
    )

    return fig, stock_data

# Streamlit UI
st.set_page_config(page_title="FinGPT Stock Analyzer", layout="wide")

st.title("FinGPT Stock Analyzer")

st.markdown("""
    <style>
    .big-font {
        font-size:20px !important;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<p class="big-font">Enter your OpenAI API Key and ask for financial data using natural language!</p>', unsafe_allow_html=True)

api_key = st.text_input("Enter your OpenAI API Key", type="password")
user_input = st.text_input("What stock data would you like to see?", "Give me the stock of Apple in the past 70 days")

if st.button("Analyze"):
    if api_key and user_input:
        with st.spinner("Analyzing..."):
            result = plot_stock_gpt(api_key, user_input)
            if isinstance(result, str):
                st.error(result)
            else:
                fig, data = result
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("Recent Data")
                st.dataframe(data.tail())

                st.subheader("Summary Statistics")
                st.dataframe(data.describe())
    else:
        st.warning("Please enter both API Key and your request.")

st.sidebar.header("About FinGPT")
st.sidebar.info("""
    FinGPT is an AI-powered stock analysis tool that uses natural language processing to interpret your requests 
    and provide relevant stock market data. It combines the power of OpenAI's language models with real-time 
    financial data to give you insights into stock performance.
""")

st.sidebar.header("How to Use")
st.sidebar.markdown("""
    1. Enter your OpenAI API Key
    2. Type your request in natural language (e.g., "Show me Tesla stock for the last 30 days")
    3. Click 'Analyze' to see the results
""")

st.sidebar.header("Example Queries")
st.sidebar.markdown("""
    - "Give me the stock of Apple in the past 70 days"
    - "Show Microsoft stock performance for the last month"
    - "What's the trend for Amazon stock in the previous quarter?"
""")