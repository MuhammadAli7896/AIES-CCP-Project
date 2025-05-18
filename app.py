import os
import sys
import traceback
import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from FinancialSentimentForecaster import FinancialSentimentForecaster

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


def initialize_analyzer():
    api_key = os.getenv("GROQ_API_KEY") or "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"
    return FinancialSentimentForecaster(groq_api_key=api_key)


ANALYZER = initialize_analyzer()


def clear_outputs():
    plt.close('all')
    return None, "", "", "", "", ""


def analyze_stock(ticker, period):
    if not ticker or not isinstance(ticker, str):
        return None, "Error: Invalid ticker input", None, None, None, "Please enter a valid stock ticker"

    if not ANALYZER:
        return None, "Error: Financial Analyzer not initialized", None, None, None, "Initialization failed"

    try:
        period_mapping = {
            "3 months": "3mo",
            "6 months": "6mo",
            "1 year": "1y",
            "2 years": "2y",
            "5 years": "5y",
            "10 years": "10y"
        }
        period_str = period_mapping.get(period, "2y")
        result = ANALYZER.analyze_asset(ticker.upper(), period=period_str)

        plt.close('all')
        fig = plt.figure(figsize=(15, 22))
        gs = fig.add_gridspec(4, 1, height_ratios=[3, 2, 2, 1.5])
        axs = [fig.add_subplot(gs[i]) for i in range(4)]

        data = result["historical_data"]
        prophet_forecast = result["prophet_forecast"]
        lstm_forecast = result["lstm_forecast"]
        recommendation = result["recommendation"]

        axs[0].plot(data.index, data['Close'], label='Close Price', color='blue')
        axs[0].plot(data.index, data['SMA20'], label='20-Day SMA', color='red', alpha=0.7)
        axs[0].plot(data.index, data['SMA50'], label='50-Day SMA', color='green', alpha=0.7)
        axs[0].plot(data.index, data['SMA200'], label='200-Day SMA', color='purple', alpha=0.7)

        last_date = data.index[-1]
        if 'predicted_prices' in prophet_forecast and not prophet_forecast['predicted_prices'].empty:
            forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=len(prophet_forecast['predicted_prices']))
            axs[0].plot(forecast_dates, prophet_forecast['predicted_prices']['yhat'].values,
                        label='Prophet Forecast', color='orange', linestyle='--')

        if 'forecast_df' in lstm_forecast and not lstm_forecast['forecast_df'].empty:
            axs[0].plot(lstm_forecast['forecast_df'].index, lstm_forecast['forecast_df']['Predicted_Close'],
                        label='LSTM Forecast', color='brown', linestyle='--')

        axs[0].set_title(f'{ticker} Price History and Forecasts ({period})', fontsize=16)
        axs[0].set_ylabel('Price (USD)', fontsize=12)
        axs[0].legend()
        axs[0].grid(True, alpha=0.3)

        axs[1].plot(data.index, data['RSI'], color='blue')
        axs[1].axhline(y=70, color='red', linestyle='--', alpha=0.5)
        axs[1].axhline(y=30, color='green', linestyle='--', alpha=0.5)
        axs[1].fill_between(data.index, y1=70, y2=100, color='red', alpha=0.1)
        axs[1].fill_between(data.index, y1=0, y2=30, color='green', alpha=0.1)
        axs[1].set_title('Relative Strength Index (RSI)', fontsize=16)
        axs[1].set_ylabel('RSI', fontsize=12)
        axs[1].set_ylim(0, 100)
        axs[1].grid(True, alpha=0.3)

        axs[2].plot(data.index, data['MACD'], label='MACD', color='blue')
        axs[2].plot(data.index, data['MACD_signal'], label='Signal Line', color='red')
        axs[2].bar(data.index, data['MACD'] - data['MACD_signal'],
                   color=np.where(data['MACD'] >= data['MACD_signal'], 'green', 'red'),
                   alpha=0.5, label='Histogram')
        axs[2].set_title('MACD', fontsize=16)
        axs[2].legend()
        axs[2].grid(True, alpha=0.3)

        rec_type = recommendation.get("type", "HOLD")
        confidence = recommendation.get("confidence", "MEDIUM")
        summary = recommendation.get("summary", "No summary available")

        bg_color = {
            "BUY": "lightgreen",
            "SELL": "lightcoral",
            "HOLD": "lightyellow"
        }.get(rec_type, "white")

        axs[3].set_facecolor(bg_color)
        axs[3].text(0.5, 0.8, f"RECOMMENDATION: {rec_type} ({confidence} CONFIDENCE)",
                    horizontalalignment='center', fontsize=14, fontweight='bold')
        axs[3].text(0.5, 0.5, summary, horizontalalignment='center',
                    verticalalignment='center', fontsize=12, wrap=True)
        axs[3].set_title('Investment Recommendation', fontsize=16)
        axs[3].axis('off')

        for ax in axs[:3]:
            ax.set_xlabel('Date', fontsize=12)

        plt.tight_layout()

        rec_text = recommendation.get("recommendation_text", "No recommendation available")
        rec_summary = recommendation.get("summary", "No summary available")

        sentiment = result["sentiment_analysis"]
        sentiment_details = (
            f"Sentiment Category: {sentiment['sentiment_category']}\n"
            f"Compound Score: {sentiment['vader_sentiment']['compound']:.2f}\n"
            f"TextBlob Polarity: {sentiment['textblob_sentiment']['polarity']:.2f}\n"
            f"TextBlob Subjectivity: {sentiment['textblob_sentiment']['subjectivity']:.2f}"
        )

        trend = result["trend_analysis"]
        trend_details = (
            f"Short-term Trend: {trend['short_term_trend']}\n"
            f"Medium-term Trend: {trend['medium_term_trend']}\n"
            f"Long-term Trend: {trend['long_term_trend']}\n"
            f"RSI: {trend['rsi_value']:.2f} ({trend['rsi_condition']})\n"
            f"Volatility: {trend['volatility']:.2f}"
        )

        return fig, rec_text, rec_summary, sentiment_details, trend_details, None

    except Exception as e:
        error_message = f"Unexpected error analyzing {ticker}: {str(e)}"
        print(error_message)
        print(traceback.format_exc())
        return None, error_message, None, None, None, error_message


def create_stock_analysis_interface():
    with gr.Blocks(title="Financial Sentiment & Forecast Analyzer") as demo:
        gr.Markdown("# 📈 Financial Sentiment & Forecast Analyzer")
        error_output = gr.Textbox(label="Error Messages", visible=False, interactive=False)

        with gr.Row():
            ticker_input = gr.Textbox(label="Enter Stock Ticker", placeholder="e.g., AAPL, GOOGL, MSFT")
            period_dropdown = gr.Dropdown(
                label="Analysis Period",
                choices=["3 months", "6 months", "1 year", "2 years", "5 years", "10 years"],
                value="2 years"
            )
            analyze_btn = gr.Button("Analyze Stock", variant="primary")

        with gr.Tabs():
            with gr.TabItem("Visualization"):
                plot_output = gr.Plot(label="Stock Analysis Visualization")
            with gr.TabItem("Recommendation"):
                rec_text_output = gr.Textbox(label="Detailed Recommendation", lines=10)
                rec_summary_output = gr.Textbox(label="Recommendation Summary")
            with gr.TabItem("Sentiment Analysis"):
                sentiment_output = gr.Textbox(label="Sentiment Details", lines=5)
            with gr.TabItem("Trend Analysis"):
                trend_output = gr.Textbox(label="Trend Details", lines=5)

        analyze_btn.click(
            fn=clear_outputs,
            inputs=[],
            outputs=[plot_output, rec_text_output, rec_summary_output,
                     sentiment_output, trend_output, error_output]
        ).then(
            fn=analyze_stock,
            inputs=[ticker_input, period_dropdown],
            outputs=[plot_output, rec_text_output,
                     rec_summary_output, sentiment_output, trend_output, error_output]
        )

        error_output.change(
            fn=lambda x: gr.update(visible=bool(x)),
            inputs=error_output,
            outputs=error_output
        )

    return demo


if __name__ == "__main__":
    try:
        demo = create_stock_analysis_interface()
        demo.launch(server_port=7860, inbrowser=True)
    except Exception as e:
        print(f"Critical error during application launch: {e}")
        traceback.print_exc()
        sys.exit(1)
