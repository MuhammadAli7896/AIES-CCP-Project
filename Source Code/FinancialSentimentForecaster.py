import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from phi.assistant import Assistant
from phi.llm.groq import Groq
from phi.tools.googlesearch import GoogleSearch
import yfinance as yf
from phi.tools.yfinance import YFinanceTools
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import warnings
import nltk

warnings.filterwarnings('ignore')
nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)


class FinancialSentimentForecaster:
    def __init__(self, groq_api_key: str = None):
        """
        Initialize the Financial Sentiment and Forecasting system
        
        Args:
            groq_api_key: API key for Groq LLM service
        """
        
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        if not self.groq_api_key:
            raise ValueError(
                "GROQ_API_KEY must be provided or set as an environment variable")

        self.sentiment_analyzer = SentimentIntensityAnalyzer()

        self.llm = Groq(
            model="llama3-70b-8192",
            api_key=self.groq_api_key,
            temperature=0.2,
        )

        self.search_tool = GoogleSearch()
        self.yfinance_tools = YFinanceTools()

        self.assistant = Assistant(
            name="Financial Analyst",
            llm=self.llm,
            tools=[
                self.search_tool,
                self.yfinance_tools
            ],
            system_prompt="""
            You are a professional financial analyst assistant. 
            Use the provided tools to gather financial data and news.
            Analyze market sentiment based on news articles and social media data.
            Identify market trends in different timeframes (short-term, medium-term, long-term).
            Provide investment recommendations with clear reasoning.
            Always consider both technical and fundamental analysis.
            """
        )

    def search_news(self, asset: str, num_results: int = 20) -> List[Dict]:
        """
        Search for recent news about a specific asset using Google Search
        
        Args:
            asset: The asset to search news for (e.g., "AAPL", "Bitcoin")
            num_results: Number of search results to return
            
        Returns:
            List of dictionaries containing news information
        """
        
        queries = [
            f"{asset} stock news recent",
            f"{asset} financial news analysis",
            f"{asset} market sentiment",
            f"{asset} price forecast",
            f"{asset} trading outlook"
        ]

        results = []
        for query in queries:
            search_results = self.search_tool.google_search(
                query, max_results=num_results//len(queries))
            results.extend(search_results)

        return results

    def analyze_news_sentiment(self, news_results: List[Dict]) -> Dict[str, Any]:
        """
        Analyze sentiment from news search results
        
        Args:
            news_results: List of news search results
            
        Returns:
            Dictionary with sentiment analysis results
        """
        
        texts = []
        for result in news_results:
            if 'snippet' in result:
                texts.append(result['snippet'])
            if 'title' in result:
                texts.append(result['title'])

        combined_text = " ".join(texts)

        vader_sentiment = self.sentiment_analyzer.polarity_scores(
            combined_text)

        textblob_sentiment = TextBlob(combined_text).sentiment

        prompt = f"""
        Analyze the market sentiment for the following financial news snippets:
        
        {combined_text[:3000]}
        
        Provide a detailed sentiment analysis including:
        1. Overall sentiment (bullish, bearish, or neutral)
        2. Key positive factors mentioned
        3. Key negative factors mentioned
        4. Any notable market concerns
        5. Any significant upcoming events mentioned
        6. Confidence level in the sentiment assessment (low, medium, high)
        """

        llm_sentiment = self.assistant.run(prompt)

        return {
            "vader_sentiment": vader_sentiment,
            "textblob_sentiment": {
                "polarity": textblob_sentiment.polarity,
                "subjectivity": textblob_sentiment.subjectivity
            },
            "llm_sentiment": llm_sentiment,
            "compound_score": vader_sentiment['compound'],
            "sentiment_category": "positive" if vader_sentiment['compound'] > 0.05 else
            "negative" if vader_sentiment['compound'] < -0.05 else "neutral"
        }

    def get_stock_data(self, ticker: str, period: str = "2y") -> pd.DataFrame:
        """
        Get historical stock data for a given ticker
        
        Args:
            ticker: Stock ticker symbol
            period: Time period for historical data (e.g., "1y", "2y", "5y")
            
        Returns:
            DataFrame with historical stock data
        """
        stock = yf.Ticker(ticker)
        stock_history = stock.history(period=period)
        return stock_history

    def get_stock_info(self, ticker: str) -> Dict:
        """
        Get stock information for a given ticker
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with stock information
        """
        
        stock = yf.Ticker(ticker)
        return stock.info

    def forecast_with_prophet(self, data: pd.DataFrame, periods: int = 30) -> Dict[str, Any]:
        """
        Forecast stock prices using Facebook Prophet
        
        Args:
            data: DataFrame with historical stock data
            periods: Number of days to forecast
            
        Returns:
            Dictionary with forecast results
        """
        try:
            df_prophet = data.reset_index()[['Date', 'Close']].rename(
                columns={'Date': 'ds', 'Close': 'y'}
            )

            df_prophet['ds'] = df_prophet['ds'].dt.tz_localize(None)

            model = Prophet(daily_seasonality=True)
            model.fit(df_prophet)

            future = model.make_future_dataframe(periods=periods)
            forecast = model.predict(future)

            components = model.plot_components(forecast)

            return {
                "forecast_df": forecast,
                "forecast_plot": model.plot(forecast),
                "components_plot": components,
                "predicted_trend": forecast[['ds', 'trend']].tail(periods),
                "predicted_prices": forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)
            }
        except Exception as e:
            print(f"Prophet forecasting failed: {e}")
            return {
                "forecast_df": pd.DataFrame(),
                "predicted_prices": pd.DataFrame({'ds': [], 'yhat': [], 'yhat_lower': [], 'yhat_upper': []}),
                "error": str(e)
            }

    def forecast_with_lstm(self, data: pd.DataFrame, prediction_days: int = 60, future_days: int = 30) -> Dict[str, Any]:
        """
        Forecast stock prices using LSTM neural network
        
        Args:
            data: DataFrame with historical stock data
            prediction_days: Number of previous days to use for prediction
            future_days: Number of days to forecast into the future
            
        Returns:
            Dictionary with forecast results
        """
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

        x_train = []
        y_train = []

        for i in range(prediction_days, len(scaled_data)):
            x_train.append(scaled_data[i-prediction_days:i, 0])
            y_train.append(scaled_data[i, 0])

        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True,
                  input_shape=(x_train.shape[1], 1)))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dense(units=25))
        model.add(Dense(units=1))

        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(x_train, y_train, epochs=25, batch_size=32, verbose=0)

        test_data = scaled_data[-prediction_days:]
        x_test = []
        x_test.append(test_data)
        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        predictions = []
        current_batch = x_test[0]

        for _ in range(future_days):
            current_pred = model.predict(
                current_batch.reshape(1, prediction_days, 1))[0]
            predictions.append(current_pred[0])
            current_batch = np.append(
                current_batch[1:], [[current_pred[0]]], axis=0)

        predicted_prices = scaler.inverse_transform(
            np.array(predictions).reshape(-1, 1))

        last_date = data.index[-1]
        future_dates = [last_date +
                        timedelta(days=i+1) for i in range(future_days)]

        forecast_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted_Close': predicted_prices.flatten()
        })
        forecast_df.set_index('Date', inplace=True)

        return {
            "forecast_df": forecast_df,
            "model": model,
            "scaled_data": scaled_data,
            "scaler": scaler
        }

    def identify_trends(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Identify market trends from historical data
        
        Args:
            data: DataFrame with historical stock data
            
        Returns:
            Dictionary with trend analysis
        """
        data['SMA20'] = data['Close'].rolling(window=20).mean()
        data['SMA50'] = data['Close'].rolling(window=50).mean()
        data['SMA200'] = data['Close'].rolling(window=200).mean()

        data['EMA12'] = data['Close'].ewm(span=12, adjust=False).mean()
        data['EMA26'] = data['Close'].ewm(span=26, adjust=False).mean()

        data['MACD'] = data['EMA12'] - data['EMA26']
        data['MACD_signal'] = data['MACD'].ewm(span=9, adjust=False).mean()

        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))

        data['BB_middle'] = data['Close'].rolling(window=20).mean()
        data['BB_upper'] = data['BB_middle'] + 2 * \
            data['Close'].rolling(window=20).std()
        data['BB_lower'] = data['BB_middle'] - 2 * \
            data['Close'].rolling(window=20).std()

        recent_data = data.tail(50).copy()

        short_term_direction = "bullish" if recent_data[
            'SMA20'].iloc[-1] > recent_data['SMA20'].iloc[-10] else "bearish"

        medium_term_direction = "bullish" if recent_data[
            'SMA50'].iloc[-1] > recent_data['SMA50'].iloc[-20] else "bearish"

        long_term_direction = "bullish" if recent_data[
            'SMA200'].iloc[-1] > recent_data['SMA200'].iloc[-50] else "bearish"

        macd_trend = "bullish" if recent_data['MACD'].iloc[-1] > recent_data['MACD_signal'].iloc[-1] else "bearish"

        latest_rsi = recent_data['RSI'].iloc[-1]
        rsi_condition = "overbought" if latest_rsi > 70 else "oversold" if latest_rsi < 30 else "neutral"

        has_golden_cross = False
        has_death_cross = False

        for i in range(1, min(20, len(recent_data))):
            if (recent_data['SMA50'].iloc[-i-1] > recent_data['SMA200'].iloc[-i-1] and
                    recent_data['SMA50'].iloc[-i] <= recent_data['SMA200'].iloc[-i]):
                has_golden_cross = True
                break
            elif (recent_data['SMA50'].iloc[-i-1] < recent_data['SMA200'].iloc[-i-1] and
                  recent_data['SMA50'].iloc[-i] >= recent_data['SMA200'].iloc[-i]):
                has_death_cross = True
                break

        data['Returns'] = data['Close'].pct_change()
        volatility = data['Returns'].std() * (252 ** 0.5)

        return {
            "short_term_trend": short_term_direction,
            "medium_term_trend": medium_term_direction,
            "long_term_trend": long_term_direction,
            "macd_trend": macd_trend,
            "rsi_value": latest_rsi,
            "rsi_condition": rsi_condition,
            "has_golden_cross": has_golden_cross,
            "has_death_cross": has_death_cross,
            "volatility": volatility,
            "technical_data": data
        }

    def get_investment_recommendation(self, ticker: str, sentiment_analysis: Dict, trend_analysis: Dict,
                                      prophet_forecast: Dict, lstm_forecast: Dict) -> Dict[str, Any]:
        """
        Generate investment recommendations based on analysis
        
        Args:
            ticker: Stock ticker symbol
            sentiment_analysis: Results from sentiment analysis
            trend_analysis: Results from trend analysis
            prophet_forecast: Results from Prophet forecast
            lstm_forecast: Results from LSTM forecast
            
        Returns:
            Dictionary with investment recommendation details
        """
        stock_info = self.get_stock_info(ticker)

        analysis_summary = f"""
        Ticker: {ticker}
        
        SENTIMENT ANALYSIS:
        Overall sentiment category: {sentiment_analysis['sentiment_category']}
        Compound sentiment score: {sentiment_analysis['vader_sentiment']['compound']}
        TextBlob polarity: {sentiment_analysis['textblob_sentiment']['polarity']}
        TextBlob subjectivity: {sentiment_analysis['textblob_sentiment']['subjectivity']}
        
        TECHNICAL ANALYSIS:
        Short-term trend: {trend_analysis['short_term_trend']}
        Medium-term trend: {trend_analysis['medium_term_trend']}
        Long-term trend: {trend_analysis['long_term_trend']}
        MACD trend: {trend_analysis['macd_trend']}
        RSI value: {trend_analysis['rsi_value']} ({trend_analysis['rsi_condition']})
        Golden Cross detected: {trend_analysis['has_golden_cross']}
        Death Cross detected: {trend_analysis['has_death_cross']}
        Volatility (annualized): {trend_analysis['volatility']}
        
        FORECASTS:
        Prophet forecast (next 30 days):
        - Start: ${prophet_forecast.get('predicted_prices', {}).get('yhat', pd.Series([0])).iloc[0]:.2f}
        - End: ${prophet_forecast.get('predicted_prices', {}).get('yhat', pd.Series([0])).iloc[-1] if not prophet_forecast.get('predicted_prices', {}).get('yhat', pd.Series([0])).empty else 0:.2f}
        - Change: {((prophet_forecast.get('predicted_prices', {}).get('yhat', pd.Series([0])).iloc[-1] / prophet_forecast.get('predicted_prices', {}).get('yhat', pd.Series([0])).iloc[0]) - 1) * 100 if not prophet_forecast.get('predicted_prices', {}).get('yhat', pd.Series([0])).empty and prophet_forecast.get('predicted_prices', {}).get('yhat', pd.Series([0])).iloc[0] != 0 else 0:.2f}%
        
        LSTM forecast (next 30 days):
        - Start: ${lstm_forecast.get('forecast_df', {}).get('Predicted_Close', pd.Series([0])).iloc[0] if not lstm_forecast.get('forecast_df', {}).get('Predicted_Close', pd.Series([0])).empty else 0:.2f}
        - End: ${lstm_forecast.get('forecast_df', {}).get('Predicted_Close', pd.Series([0])).iloc[-1] if not lstm_forecast.get('forecast_df', {}).get('Predicted_Close', pd.Series([0])).empty else 0:.2f}
        - Change: {((lstm_forecast.get('forecast_df', {}).get('Predicted_Close', pd.Series([0])).iloc[-1] / lstm_forecast.get('forecast_df', {}).get('Predicted_Close', pd.Series([0])).iloc[0]) - 1) * 100 if not lstm_forecast.get('forecast_df', {}).get('Predicted_Close', pd.Series([0])).empty and lstm_forecast.get('forecast_df', {}).get('Predicted_Close', pd.Series([0])).iloc[0] != 0 else 0:.2f}%
        
        FUNDAMENTAL INFORMATION:
        Company: {stock_info.get('longName', ticker)}
        Sector: {stock_info.get('sector', 'N/A')}
        Industry: {stock_info.get('industry', 'N/A')}
        Market Cap: ${stock_info.get('marketCap', 0) / 1_000_000_000:.2f} billion
        P/E Ratio: {stock_info.get('trailingPE', 'N/A')}
        Forward P/E: {stock_info.get('forwardPE', 'N/A')}
        PEG Ratio: {stock_info.get('pegRatio', 'N/A')}
        Dividend Yield: {stock_info.get('dividendYield', 0) * 100:.2f}%
        52-Week High: ${stock_info.get('fiftyTwoWeekHigh', 0):.2f}
        52-Week Low: ${stock_info.get('fiftyTwoWeekLow', 0):.2f}
        """

        prompt = f"""
        Based on the following comprehensive analysis of {ticker}, provide an investment recommendation.
        
        {analysis_summary}
        
        In your recommendation:
        1. Start with a clear "RECOMMENDATION SUMMARY" section that provides a concise 1-2 sentence summary of your overall recommendation (BUY/SELL/HOLD with confidence level and timeframe)
        2. Determine if this is a good buy for short-term trading (days to weeks)
        3. Determine if this is a good buy for long-term holding (months to years)
        4. List the primary factors supporting your recommendation
        5. List potential risks that could change your outlook
        6. Suggest an appropriate position sizing based on the risk profile
        7. Include any relevant price targets or stop-loss recommendations
        
        Format your response with clear section headers and bullet points where appropriate.
        Your analysis should combine technical, fundamental, and sentiment factors.
        """

        recommendation_text = ""
        for chunk in self.assistant.run(prompt):
            recommendation_text += chunk

        lines = recommendation_text.split('\n')
        summary_line = ""
        for line in lines:
            if "RECOMMENDATION SUMMARY" in line:
                summary_index = lines.index(line) + 1
                if summary_index < len(lines):
                    summary_line = lines[summary_index].strip()
                    break

        rec_type = "HOLD"  
        if "BUY" in recommendation_text.upper() or "BULLISH" in recommendation_text.upper():
            rec_type = "BUY"
        elif "SELL" in recommendation_text.upper() or "BEARISH" in recommendation_text.upper():
            rec_type = "SELL"

        confidence = "MEDIUM"  
        if "HIGH CONFIDENCE" in recommendation_text.upper() or "STRONG" in recommendation_text.upper():
            confidence = "HIGH"
        elif "LOW CONFIDENCE" in recommendation_text.upper() or "CAUTIOUS" in recommendation_text.upper():
            confidence = "LOW"

        return {
            "recommendation_text": recommendation_text,
            "summary": summary_line if summary_line else "No summary available",
            "type": rec_type,
            "confidence": confidence
        }
    
    def analyze_asset(self, ticker: str, period: str = "2y") -> Dict[str, Any]:
        """
        Run a complete analysis of an asset
        
        Args:
            ticker: Stock ticker symbol
            period: Time period for historical data
            
        Returns:
            Dictionary with complete analysis results
        """
        print(f"Analyzing {ticker}...")
        print("Gathering news and analyzing sentiment...")
        news_results = self.search_news(ticker)
        sentiment_analysis = self.analyze_news_sentiment(news_results)

        print("Getting historical data...")
        stock_data = self.get_stock_data(ticker, period)

        print("Identifying trends...")
        trend_analysis = self.identify_trends(stock_data)

        print("Generating forecasts with Prophet...")
        try:
            prophet_forecast = self.forecast_with_prophet(stock_data)
        except Exception as e:
            print(f"Prophet forecasting failed: {e}")
            prophet_forecast = {"error": str(e)}

        print("Generating forecasts with LSTM...")
        try:
            lstm_forecast = self.forecast_with_lstm(stock_data)
        except Exception as e:
            print(f"LSTM forecasting failed: {e}")
            lstm_forecast = {"error": str(e)}

        print("Formulating investment recommendation...")
        recommendation = self.get_investment_recommendation(
            ticker, sentiment_analysis, trend_analysis, prophet_forecast, lstm_forecast
        )

        result = {
            "ticker": ticker,
            "analysis_date": datetime.now().strftime("%Y-%m-%d"),
            "sentiment_analysis": sentiment_analysis,
            "trend_analysis": trend_analysis,
            "prophet_forecast": prophet_forecast,
            "lstm_forecast": lstm_forecast,
            "recommendation": recommendation,
            "historical_data": stock_data
        }

        return result