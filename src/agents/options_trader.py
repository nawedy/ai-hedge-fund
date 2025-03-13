from langchain_openai import ChatOpenAI
from graph.state import AgentState, show_agent_reasoning
from tools.api import get_option_metrics, get_market_cap, search_line_items
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
import json
from typing_extensions import Literal
from utils.progress import progress
from utils.llm import call_llm

class OptionsTraderSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float
    reasoning: str

def options_trader_agent(state: AgentState):
    """
    Analyzes options data and generates trading signals for options trading.
    """
    data = state["data"]
    end_date = data["end_date"]
    tickers = data["tickers"]
    
    analysis_data = {}
    options_analysis = {}
    
    for ticker in tickers:
        progress.update_status("options_trader_agent", ticker, "Fetching option metrics")
        metrics = get_option_metrics(ticker, end_date)
        
        progress.update_status("options_trader_agent", ticker, "Gathering financial line items")
        financial_line_items = search_line_items(
            ticker,
            ["volatility", "open_interest", "implied_volatility"],
            end_date,
            period="daily",  # Daily data for options trading
            limit=30         # fetch up to 30 days of data
        )
        
        progress.update_status("options_trader_agent", ticker, "Analyzing options data")
        analysis = analyze_options_data(metrics, financial_line_items)
        
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
        
        progress.update_status("options_trader_agent", ticker, "Generating Options Trader analysis")
        options_output = generate_options_output(
            ticker=ticker, 
            analysis_data=analysis_data,
            model_name=state["metadata"]["model_name"],
            model_provider=state["metadata"]["model_provider"],
        )
        
        options_analysis[ticker] = {
            "signal": options_output.signal,
            "confidence": options_output.confidence,
            "reasoning": options_output.reasoning
        }
        
        progress.update_status("options_trader_agent", ticker, "Done")
    
    message = HumanMessage(
        content=json.dumps(options_analysis),
        name="options_trader_agent"
    )
    
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(options_analysis, "Options Trader Agent")
    
    state["data"]["analyst_signals"]["options_trader_agent"] = options_analysis

    return {
        "messages": [message],
        "data": state["data"]
    }

def analyze_options_data(metrics: list, financial_line_items: list) -> dict:
    """
    Analyze options data to generate a score and reasoning for trading signals.
    """
    score = 0
    details = []
    
    if not metrics or not financial_line_items:
        return {
            "score": 0,
            "details": "Insufficient data to analyze options data"
        }
    
    # Example analysis (adjust as needed):
    volatility = [item.volatility for item in financial_line_items if item.volatility is not None]
    if volatility and max(volatility) > 0.3:
        score += 2
        details.append("High volatility indicates potential trading opportunities.")
    else:
        details.append("Volatility is low.")
    
    open_interest = [item.open_interest for item in financial_line_items if item.open_interest is not None]
    if open_interest and max(open_interest) > 1000:
        score += 2
        details.append("High open interest indicates strong market activity.")
    else:
        details.append("Open interest is low.")
    
    implied_volatility = [item.implied_volatility for item in financial_line_items if item.implied_volatility is not None]
    if implied_volatility and max(implied_volatility) > 0.25:
        score += 1
        details.append("High implied volatility suggests significant market expectations.")
    else:
        details.append("Implied volatility is low.")
    
    return {
        "score": score,
        "details": "; ".join(details)
    }

def generate_options_output(
    ticker: str,
    analysis_data: dict[str, any],
    model_name: str,
    model_provider: str,
) -> OptionsTraderSignal:
    """
    Generates trading decisions for options trading.
    """
    template = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an Options Trader AI agent, making trading decisions based on options data.
            
            1. Analyze volatility, open interest, and implied volatility.
            2. Generate trading signals (bullish, bearish, neutral) with confidence levels.
            3. Provide a data-driven reasoning for the signals."""
        ),
        (
            "human",
            """Based on the following analysis, create an Options Trader signal.
            
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

    def create_default_options_trader_signal():
        return OptionsTraderSignal(
            signal="neutral",
            confidence=0.0,
            reasoning="Error in analysis, defaulting to neutral"
        )

    return call_llm(
        prompt=prompt, 
        model_name=model_name, 
        model_provider=model_provider, 
        pydantic_model=OptionsTraderSignal, 
        agent_name="options_trader_agent", 
        default_factory=create_default_options_trader_signal,
    )
