from langchain_openai import ChatOpenAI
from graph.state import AgentState, show_agent_reasoning
from tools.api import get_swing_trade_metrics, get_market_cap, search_line_items
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
import json
from typing_extensions import Literal
from utils.progress import progress
from utils.llm import call_llm

class SwingTraderSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float
    reasoning: str

def swing_trader_agent(state: AgentState):
    """
    Analyzes stock data over a few days to generate swing trading signals.
    """
    data = state["data"]
    end_date = data["end_date"]
    tickers = data["tickers"]
    
    analysis_data = {}
    swing_trader_analysis = {}
    
    for ticker in tickers:
        progress.update_status("swing_trader_agent", ticker, "Fetching swing trade metrics")
        metrics = get_swing_trade_metrics(ticker, end_date)
        
        progress.update_status("swing_trader_agent", ticker, "Gathering financial line items")
        financial_line_items = search_line_items(
            ticker,
            ["price_movement", "volume", "volatility"],
            end_date,
            period="daily",  # Daily data for swing trading
            limit=7          # fetch up to 7 days of data
        )
        
        progress.update_status("swing_trader_agent", ticker, "Analyzing swing trade data")
        analysis = analyze_swing_trade_data(metrics, financial_line_items)
        
        total_score = analysis["score"]
        max_possible_score = 10  # Adjust weighting as desired
        
        if total_score >= 0.7 * max_possible_score:
            signal = "bullish"
        elif total_score <= 0.3 * max_possible_score:
            signal = "bearish"
        else:
            signal = "neutral"
        
        analysis_data[ticker] = {
            "signal": signal,
            "score": total_score,
            "max_score": max_possible_score,
            "analysis": analysis
        }
        
        progress.update_status("swing_trader_agent", ticker, "Generating Swing Trader analysis")
        swing_trader_output = generate_swing_trader_output(
            ticker=ticker, 
            analysis_data=analysis_data,
            model_name=state["metadata"]["model_name"],
            model_provider=state["metadata"]["model_provider"],
        )
        
        swing_trader_analysis[ticker] = {
            "signal": swing_trader_output.signal,
            "confidence": swing_trader_output.confidence,
            "reasoning": swing_trader_output.reasoning
        }
        
        progress.update_status("swing_trader_agent", ticker, "Done")
    
    message = HumanMessage(
        content=json.dumps(swing_trader_analysis),
        name="swing_trader_agent"
    )
    
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(swing_trader_analysis, "Swing Trader Agent")
    
    state["data"]["analyst_signals"]["swing_trader_agent"] = swing_trader_analysis

    return {
        "messages": [message],
        "data": state["data"]
    }

def analyze_swing_trade_data(metrics: list, financial_line_items: list) -> dict:
    """
    Analyze data to generate a score and reasoning for swing trading signals.
    """
    score = 0
    details = []
    
    if not metrics or not financial_line_items:
        return {
            "score": 0,
            "details": "Insufficient data to analyze swing trade data"
        }
    
    # Example analysis (adjust as needed):
    price_movement = [item.price_movement for item in financial_line_items if item.price_movement is not None]
    if price_movement and abs(price_movement[-1] - price_movement[0]) > 5.0:
        score += 2
        details.append("Significant price movement over the week indicates potential swing trading opportunities.")
    else:
        details.append("Price movement over the week is not significant.")
    
    volume = [item.volume for item in financial_line_items if item.volume is not None]
    if volume and max(volume) > 500000:
        score += 2
        details.append("High trading volume indicates strong market activity.")
    else:
        details.append("Trading volume is low.")
    
    volatility = [item.volatility for item in financial_line_items if item.volatility is not None]
    if volatility and max(volatility) > 0.05:
        score += 1
        details.append("High volatility suggests significant market fluctuations.")
    else:
        details.append("Volatility is low.")
    
    return {
        "score": score,
        "details": "; ".join(details)
    }

def generate_swing_trader_output(
    ticker: str,
    analysis_data: dict[str, any],
    model_name: str,
    model_provider: str,
) -> SwingTraderSignal:
    """
    Generates trading decisions for swing trading.
    """
    template = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a Swing Trader AI agent, making trading decisions based on short-term data.
            
            1. Analyze price movement, volume, and volatility over a few days.
            2. Generate trading signals (bullish, bearish, neutral) with confidence levels.
            3. Provide a data-driven reasoning for the signals."""
        ),
        (
            "human",
            """Based on the following analysis, create a Swing Trader signal.
            
            Analysis Data for {ticker}:
            {analysis_data}
            
            Return the trading signal in this JSON format:
            {{
              "signal": "bullish/bearish/neutral",
              "confidence": float (0-100),
              "reasoning": "string"
            }}
            """
        )
    ])

    prompt = template.invoke({
        "analysis_data": json.dumps(analysis_data, indent=2),
        "ticker": ticker
    })

    def create_default_swing_trader_signal
