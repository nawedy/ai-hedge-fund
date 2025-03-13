from langchain_openai import ChatOpenAI
from graph.state import AgentState, show_agent_reasoning
from tools.api import get_intraday_metrics, get_market_cap, search_line_items
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
import json
from typing_extensions import Literal
from utils.progress import progress
from utils.llm import call_llm

class DayTraderSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float
    reasoning: str

def day_trader_agent(state: AgentState):
    """
    Analyzes intraday stock data to generate day trading signals.
    """
    data = state["data"]
    end_date = data["end_date"]
    tickers = data["tickers"]
    
    analysis_data = {}
    day_trader_analysis = {}
    
    for ticker in tickers:
        progress.update_status("day_trader_agent", ticker, "Fetching intraday metrics")
        metrics = get_intraday_metrics(ticker, end_date)
        
        progress.update_status("day_trader_agent", ticker, "Gathering financial line items")
        financial_line_items = search_line_items(
            ticker,
            ["price_movement", "volume", "volatility"],
            end_date,
            period="intraday",  # Intraday data for day trading
            limit=1             # fetch data for the current day
        )
        
        progress.update_status("day_trader_agent", ticker, "Analyzing intraday data")
        analysis = analyze_intraday_data(metrics, financial_line_items)
        
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
        
        progress.update_status("day_trader_agent", ticker, "Generating Day Trader analysis")
        day_trader_output = generate_day_trader_output(
            ticker=ticker, 
            analysis_data=analysis_data,
            model_name=state["metadata"]["model_name"],
            model_provider=state["metadata"]["model_provider"],
        )
        
        day_trader_analysis[ticker] = {
            "signal": day_trader_output.signal,
            "confidence": day_trader_output.confidence,
            "reasoning": day_trader_output.reasoning
        }
        
        progress.update_status("day_trader_agent", ticker, "Done")
    
    message = HumanMessage(
        content=json.dumps(day_trader_analysis),
        name="day_trader_agent"
    )
    
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(day_trader_analysis, "Day Trader Agent")
    
    state["data"]["analyst_signals"]["day_trader_agent"] = day_trader_analysis

    return {
        "messages": [message],
        "data": state["data"]
    }

def analyze_intraday_data(metrics: list, financial_line_items: list) -> dict:
    """
    Analyze intraday data to generate a score and reasoning for trading signals.
    """
    score = 0
    details = []
    
    if not metrics or not financial_line_items:
        return {
            "score": 0,
            "details": "Insufficient data to analyze intraday data"
        }
    
    # Example analysis (adjust as needed):
    price_movement = [item.price_movement for item in financial_line_items if item.price_movement is not None]
    if price_movement and abs(price_movement[0]) > 1.0:
        score += 2
        details.append("Significant price movement indicates potential trading opportunities.")
    else:
        details.append("Price movement is not significant.")
    
    volume = [item.volume for item in financial_line_items if item.volume is not None]
    if volume and max(volume) > 1000000:
        score += 2
        details.append("High trading volume indicates strong market activity.")
    else:
        details.append("Trading volume is low.")
    
    volatility = [item.volatility for item in financial_line_items if item.volatility is not None]
    if volatility and max(volatility) > 0.03:
        score += 1
        details.append("High volatility suggests significant market fluctuations.")
    else:
        details.append("Volatility is low.")
    
    return {
        "score": score,
        "details": "; ".join(details)
    }

def generate_day_trader_output(
    ticker: str,
    analysis_data: dict[str, any],
    model_name: str,
    model_provider: str,
) -> DayTraderSignal:
    """
    Generates trading decisions for day trading.
    """
    template = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a Day Trader AI agent, making trading decisions based on intraday data.
            
            1. Analyze price movement, volume, and volatility.
            2. Generate trading signals (bullish, bearish, neutral) with confidence levels.
            3. Provide a data-driven reasoning for the signals."""
        ),
        (
            "human",
            """Based on the following analysis, create a Day Trader signal.
            
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

    def create_default_day_trader_signal():
        return DayTraderSignal(
            signal="neutral",
            confidence=0.0,
            reasoning="Error in analysis, defaulting to neutral"
        )

    return call_llm(
        prompt=prompt, 
        model_name=model_name, 
        model_provider=model_provider, 
        pydantic_model=DayTraderSignal, 
        agent_name="day_trader_agent", 
        default_factory=create_default_day_trader_signal,
    )
