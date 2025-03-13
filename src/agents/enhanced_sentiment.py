from langchain_core.messages import HumanMessage
from graph.state import AgentState, show_agent_reasoning
from utils.progress import progress
import pandas as pd
import numpy as np
import json

from tools.api import get_insider_trades, get_company_news, get_twitter_sentiment, get_reddit_sentiment

##### Enhanced Sentiment Agent #####
def enhanced_sentiment_agent(state: AgentState):
    """Analyzes market sentiment including social media (Twitter/X and Reddit) and generates trading signals for multiple tickers."""
    data = state.get("data", {})
    end_date = data.get("end_date")
    tickers = data.get("tickers")

    # Initialize sentiment analysis for each ticker
    sentiment_analysis = {}

    for ticker in tickers:
        progress.update_status("enhanced_sentiment_agent", ticker, "Fetching insider trades")

        # Get the insider trades
        insider_trades = get_insider_trades(
            ticker=ticker,
            end_date=end_date,
            limit=1000,
        )

        progress.update_status("enhanced_sentiment_agent", ticker, "Analyzing trading patterns")

        # Get the signals from the insider trades
        transaction_shares = pd.Series([t.transaction_shares for t in insider_trades]).dropna()
        insider_signals = np.where(transaction_shares < 0, "bearish", "bullish").tolist()

        progress.update_status("enhanced_sentiment_agent", ticker, "Fetching company news")

        # Get the company news
        company_news = get_company_news(ticker, end_date, limit=100)

        # Get the sentiment from the company news
        sentiment = pd.Series([n.sentiment for n in company_news]).dropna()
        news_signals = np.where(sentiment == "negative", "bearish", 
                              np.where(sentiment == "positive", "bullish", "neutral")).tolist()
        
        progress.update_status("enhanced_sentiment_agent", ticker, "Fetching Twitter/X sentiment")

        # Get the Twitter/X sentiment
        twitter_sentiment = get_twitter_sentiment(ticker, end_date, limit=100)
        
        # Get the sentiment from Twitter/X
        twitter_signals = pd.Series([t.sentiment for t in twitter_sentiment]).dropna()
        twitter_signals = np.where(twitter_signals == "negative", "bearish", 
                                 np.where(twitter_signals == "positive", "bullish", "neutral")).tolist()
        
        progress.update_status("enhanced_sentiment_agent", ticker, "Fetching Reddit sentiment")

        # Get the Reddit sentiment
        reddit_sentiment = get_reddit_sentiment(ticker, end_date, limit=100)
        
        # Get the sentiment from Reddit
        reddit_signals = pd.Series([r.sentiment for r in reddit_sentiment]).dropna()
        reddit_signals = np.where(reddit_signals == "negative", "bearish", 
                                np.where(reddit_signals == "positive", "bullish", "neutral")).tolist()
        
        progress.update_status("enhanced_sentiment_agent", ticker, "Combining signals")
        
        # Combine signals from all sources with weights
        insider_weight = 0.2
        news_weight = 0.3
        twitter_weight = 0.3
        reddit_weight = 0.2
        
        # Calculate weighted signal counts
        bullish_signals = (
            insider_signals.count("bullish") * insider_weight +
            news_signals.count("bullish") * news_weight +
            twitter_signals.count("bullish") * twitter_weight +
            reddit_signals.count("bullish") * reddit_weight
        )
        bearish_signals = (
            insider_signals.count("bearish") * insider_weight +
            news_signals.count("bearish") * news_weight +
            twitter_signals.count("bearish") * twitter_weight +
            reddit_signals.count("bearish") * reddit_weight
        )

        if bullish_signals > bearish_signals:
            overall_signal = "bullish"
        elif bearish_signals > bullish_signals:
            overall_signal = "bearish"
        else:
            overall_signal = "neutral"

        # Calculate confidence level based on the weighted proportion
        total_weighted_signals = (
            len(insider_signals) * insider_weight + 
            len(news_signals) * news_weight + 
            len(twitter_signals) * twitter_weight + 
            len(reddit_signals) * reddit_weight
        )
        confidence = 0  # Default confidence when there are no signals
        if total_weighted_signals > 0:
            confidence = round(max(bullish_signals, bearish_signals) / total_weighted_signals, 2) * 100
        reasoning = f"Weighted Bullish signals: {bullish_signals:.1f}, Weighted Bearish signals: {bearish_signals:.1f}"

        sentiment_analysis[ticker] = {
            "signal": overall_signal,
            "confidence": confidence,
            "reasoning": reasoning,
        }

        progress.update_status("enhanced_sentiment_agent", ticker, "Done")

    # Create the sentiment message
    message = HumanMessage(
        content=json.dumps(sentiment_analysis),
        name="enhanced_sentiment_agent",
    )

    # Print the reasoning if the flag is set
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(sentiment_analysis, "Enhanced Sentiment Analysis Agent")

    # Add the signal to the analyst_signals list
    state["data"]["analyst_signals"]["enhanced_sentiment_agent"] = sentiment_analysis

    return {
        "messages": [message],
        "data": data,
    }
