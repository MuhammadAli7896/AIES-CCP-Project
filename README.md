# Financial Sentiment & Forecast Analyzer

A comprehensive stock market analysis tool that combines technical indicators, machine learning forecasts, and sentiment analysis to provide investment recommendations and price predictions.

##  Project Overview

The Financial Sentiment & Forecast Analyzer is a sophisticated web application built with Gradio that helps investors make informed decisions by analyzing stock market data through multiple lenses:

1. **Technical Analysis**: Evaluates price movements using SMA, RSI, MACD, and other indicators.  
2. **Machine Learning Forecasts**: Predicts future price movements using Prophet and LSTM models.  
3. **Sentiment Analysis**: Gauges market sentiment using natural language processing techniques.  
4. **Investment Recommendations**: Provides actionable insights based on combined analysis results.  

## Youtube Overview
  https://youtu.be/w5qDHLH_9Ck

##  Features

- **Interactive User Interface**: Easy-to-use Gradio interface with tabbed results display.  
- **Comprehensive Stock Analysis**: View technical indicators, forecasts, and sentiment in one dashboard.  
- **Multi-timeframe Support**: Analyze stocks over various timeframes (3 months to 10 years).  
- **Advanced Visualizations**:  
  - Price history with 20, 50, and 200-day moving averages.  
  - RSI (Relative Strength Index) with overbought/oversold indicators.  
  - MACD (Moving Average Convergence Divergence) with signal line and histogram.  
  - Visual recommendation display with color-coding.  
- **Dual Forecast Models**:  
  - Prophet model for statistical forecasting.  
  - LSTM neural network for deep learning-based predictions.  
- **NLP-Powered Sentiment Analysis**: Combines VADER and TextBlob for robust sentiment scoring.  

##  Technologies Used

- **Frontend**: Gradio (web interface framework).  
- **Data Analysis**: Pandas, NumPy.  
- **Visualization**: Matplotlib.  
- **Machine Learning**: Prophet, LSTM.  
- **Natural Language Processing**: VADER, TextBlob.  
- **API Integration**: GROQ API for enhanced language model capabilities.  

##  Requirements

This project requires the following Python packages:  

- gradio  
- pandas  
- numpy  
- matplotlib  
- prophet  
- tensorflow  
- textblob  
- vadersentiment  
- yfinance  
- scikit-learn  
- groq  

You can install all dependencies using the provided `requirements.txt` file.

##  Configuration

The application uses the GROQ API for enhanced language model capabilities. You can provide your own API key through environment variables:

For Windows:
```bash
set GROQ_API_KEY=your_api_key_here
```
For Mac and Linux:

```bash
export GROQ_API_KEY="your_api_key_here"
```

If no API key is provided, the application will use a default key (note that this key may be rate-limited).

##  Installation

```bash
# Clone the repository
git clone https://github.com/MuhammadAli7896/AIES-CCP-Project.git
cd AIES-CCP-Project
cd "Source Code"

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

##  Usage

To run the application:

```bash
python app.py
```

The web interface will automatically open in your default browser at http://localhost:7860.

### Using the Interface

1. **Enter a valid stock ticker symbol** (e.g., AAPL, MSFT, GOOGL).  
2. **Select your desired analysis timeframe.**  
3. **Click "Analyze Stock".**  
4. View results in the tabbed interface:  
   - **Visualization**: Technical indicators and price forecasts.  
   - **Recommendation**: Investment advice based on all analysis factors.  
   - **Sentiment Analysis**: Market sentiment metrics.  
   - **Trend Analysis**: Short, medium, and long-term trend indicators.  

##  How It Works

### Data Collection and Processing  
The application fetches historical stock data and processes it to calculate technical indicators like moving averages, RSI, and MACD.  

### Machine Learning Models  
- **Prophet**: A time series forecasting model developed by Facebook Research.  
- **LSTM**: A recurrent neural network architecture for sequence prediction.  

### Sentiment Analysis  
The system analyzes news and social media sentiment related to the stock using VADER and TextBlob sentiment analyzers, providing a comprehensive view of market perception.  

### Recommendation Engine  
All data points are combined to generate a final investment recommendation (**BUY**, **HOLD**, or **SELL**) with confidence levels and supporting rationale.
