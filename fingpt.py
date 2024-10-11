import gradio as gr
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from openai import OpenAI

# Preset for the OpenAI model
preset = "You now act as an agent to help me transform user's input to a formatted dictionary. The user should only request for financial stock market data. If it is not the case, you return a message 'ERROR'. If the user indeed asks for financial data, for example, give me the stock of apple in recent 60 days. Then you return a dictionary {'stock': 'AAPL', 'time': 60}.\n If no date information is found, then you set it by default to 60\n"

def generate_response(api_key, prompt, model="gpt-3.5-turbo", max_tokens=100):
    # Initialize the OpenAI client with the provided API key
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
            return gr.Error("Invalid input! Try again!")
    except:
        return gr.Error("Invalid input!")
    
    return plot_stock(input_dict['stock'], input_dict['time'])

def plot_stock(stock_name, time_period):
    # Fetch stock data
    end_date = pd.to_datetime('today')
    start_date = end_date - pd.Timedelta(days=time_period)
    stock = yf.Ticker(stock_name)
    stock_data = stock.history(start=start_date, end=end_date)

    # Plot stock data
    fig, ax = plt.subplots()
    ax.plot(stock_data['Close'])
    ax.set_xlabel('Date')
    ax.set_ylabel('Close Price')
    ax.set_title(f'{stock_name} Stock Prices for the Last {time_period} Days')
    ax.grid()

    # Save plot as an image
    fig.savefig('stock_plot.png', bbox_inches='tight')
    plt.close(fig)  # Close the figure to free up memory

    return 'stock_plot.png'

# Define Gradio input and output components
api_key_input = gr.Textbox(label="Enter your OpenAI API Key", type="password")
user_input = gr.Textbox(label="Input your requirements!")
inputs = [api_key_input, user_input]
output = gr.Image(label="Stock Price Plot")

# Create the Gradio interface
iface = gr.Interface(
    fn=lambda api_key, prompt: plot_stock_gpt(api_key, prompt),
    inputs=inputs,
    outputs=output,
    title="Stock Price Plotter",
    description="This is FinGPT! Enter your OpenAI API Key and try asking for financial data with natural language!",
    examples=[["sk-YourAPIKey", "Give me the stock of Apple in the past 70 days"]]
)

# Launch the app
iface.launch()
