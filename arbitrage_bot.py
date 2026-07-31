#!/usr/bin/env python3
"""
Micro-Arbitrage Bot for Solana
Monitors price differences between DEXs and executes profitable trades
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional, Dict, List
from datetime import datetime

import httpx
from solana.rpc.async_api import AsyncClient
from solders.compute_budget import set_compute_unit_price  # For priority fees
from solders.keypair import Keypair
from solders.pubkey import Pubkey
from solders.instruction import Instruction
from solders.system_program import TransferParams, transfer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('arbitrage_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class PriceQuote:
    """Price quote from a DEX"""
    dex: str
    input_token: str
    output_token: str
    input_amount: Decimal
    output_amount: Decimal
    price: Decimal  # output/input
    fee: Decimal
    timestamp: datetime

@dataclass
class ArbitrageOpportunity:
    """Detected arbitrage opportunity"""
    buy_dex: str
    sell_dex: str
    token_pair: str
    buy_price: Decimal
    sell_price: Decimal
    spread_percent: Decimal
    profit_after_fees: Decimal
    timestamp: datetime

class SolanaArbitrageBot:
    """
    Micro-arbitrage bot for Solana DEXs
    
    Strategy:
    1. Monitor SOL/USDC prices across Jupiter and Raydium
    2. Detect price discrepancies > threshold
    3. Execute simultaneous buy/sell (atomic where possible)
    4. Track all trades and P&L
    """
    
    def __init__(
        self,
        wallet_address: str,
        rpc_url: str = "https://api.mainnet-beta.solana.com",
        min_spread_percent: Decimal = Decimal("0.5"),
        max_trade_size_sol: Decimal = Decimal("0.01"),  # Start tiny
        max_slippage_percent: Decimal = Decimal("0.1")
    ):
        self.wallet_address = Pubkey.from_string(wallet_address)
        self.rpc_url = rpc_url
        self.min_spread_percent = min_spread_percent
        self.max_trade_size_sol = max_trade_size_sol
        self.max_slippage_percent = max_slippage_percent
        
        self.client: Optional[AsyncClient] = None
        self.price_history: List[PriceQuote] = []
        self.trade_history: List[Dict] = []
        self.total_pnl = Decimal("0")
        self.total_fees = Decimal("0")
        
        # DEX endpoints
        self.jupiter_api = "https://quote-api.jup.ag/v6"
        self.raydium_api = "https://api.raydium.io/v2"
        
        logger.info(f"Bot initialized for wallet: {wallet_address}")
        logger.info(f"Min spread: {min_spread_percent}%, Max trade: {max_trade_size_sol} SOL")
    
    async def connect(self):
        """Connect to Solana RPC"""
        self.client = AsyncClient(self.rpc_url)
        logger.info(f"Connected to Solana RPC: {self.rpc_url}")
    
    async def disconnect(self):
        """Disconnect from Solana RPC"""
        if self.client:
            await self.client.close()
            logger.info("Disconnected from Solana RPC")
    
    async def get_jupiter_quote(
        self,
        input_token: str,
        output_token: str,
        amount: Decimal
    ) -> Optional[PriceQuote]:
        """Get price quote from Jupiter"""
        try:
            # Jupiter uses lamports for SOL (1 SOL = 10^9 lamports)
            amount_lamports = int(amount * Decimal("1000000000"))
            
            url = f"{self.jupiter_api}/quote"
            params = {
                "inputMint": input_token,
                "outputMint": output_token,
                "amount": amount_lamports,
                "slippageBps": int(self.max_slippage_percent * 100)  # Convert to basis points
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.get(url, params=params, timeout=10.0)
                
                if response.status_code == 200:
                    data = response.json()
                    output_amount = Decimal(data.get("outAmount", 0)) / Decimal("1000000")  # USDC has 6 decimals
                    
                    # Calculate price (USDC per SOL)
                    price = output_amount / amount
                    
                    return PriceQuote(
                        dex="Jupiter",
                        input_token=input_token,
                        output_token=output_token,
                        input_amount=amount,
                        output_amount=output_amount,
                        price=price,
                        fee=Decimal("0.001"),  # Approximate 0.1% fee
                        timestamp=datetime.now()
                    )
                else:
                    logger.warning(f"Jupiter API error: {response.status_code}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error getting Jupiter quote: {e}")
            return None
    
    async def get_raydium_quote(
        self,
        input_token: str,
        output_token: str,
        amount: Decimal
    ) -> Optional[PriceQuote]:
        """Get price quote from Raydium"""
        try:
            # Raydium uses a different API structure
            # For now, we'll simulate with a placeholder
            # In production, this would call Raydium's swap API
            
            # TODO: Implement actual Raydium quote fetching
            # This requires getting pool info and calculating swap amounts
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting Raydium quote: {e}")
            return None
    
    async def find_arbitrage_opportunities(
        self,
        token_pair: str = "SOL/USDC"
    ) -> List[ArbitrageOpportunity]:
        """
        Find arbitrage opportunities between DEXs
        
        Returns list of opportunities sorted by profit potential
        """
        opportunities = []
        
        # Token addresses
        SOL_MINT = "So11111111111111111111111111111111111111112"
        USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
        
        # Get quotes from both DEXs
        jupiter_quote = await self.get_jupiter_quote(
            SOL_MINT, USDC_MINT, self.max_trade_size_sol
        )
        
        raydium_quote = await self.get_raydium_quote(
            SOL_MINT, USDC_MINT, self.max_trade_size_sol
        )
        
        quotes = [q for q in [jupiter_quote, raydium_quote] if q is not None]
        
        if len(quotes) < 2:
            logger.warning("Could not get quotes from multiple DEXs")
            return opportunities
        
        # Compare all pairs
        for i, buy_quote in enumerate(quotes):
            for j, sell_quote in enumerate(quotes):
                if i >= j:
                    continue
                
                # Calculate spread
                spread = sell_quote.price - buy_quote.price
                spread_percent = (spread / buy_quote.price) * 100
                
                # Estimate fees
                trading_fee = buy_quote.output_amount * Decimal("0.003")  # 0.3% typical
                network_fee = Decimal("0.000005")  # ~0.000005 SOL
                total_fees = trading_fee + network_fee
                
                # Calculate profit
                gross_profit = spread * self.max_trade_size_sol
                net_profit = gross_profit - total_fees
                
                if spread_percent >= self.min_spread_percent and net_profit > 0:
                    opportunity = ArbitrageOpportunity(
                        buy_dex=buy_quote.dex,
                        sell_dex=sell_quote.dex,
                        token_pair=token_pair,
                        buy_price=buy_quote.price,
                        sell_price=sell_quote.price,
                        spread_percent=spread_percent,
                        profit_after_fees=net_profit,
                        timestamp=datetime.now()
                    )
                    opportunities.append(opportunity)
                    
                    logger.info(f"Opportunity found: {opportunity}")
        
        # Sort by profit
        opportunities.sort(key=lambda x: x.profit_after_fees, reverse=True)
        return opportunities
    
    async def execute_arbitrage(self, opportunity: ArbitrageOpportunity) -> bool:
        """
        Execute an arbitrage trade
        
        For now, this logs the trade but doesn't execute (paper trading mode)
        In production, this would:
        1. Build buy transaction on buy_dex
        2. Build sell transaction on sell_dex
        3. Submit as bundle if possible
        4. Wait for confirmation
        """
        logger.info(f"Would execute arbitrage: {opportunity}")
        
        # Simulate trade execution
        trade_record = {
            "timestamp": datetime.now().isoformat(),
            "type": "arbitrage",
            "buy_dex": opportunity.buy_dex,
            "sell_dex": opportunity.sell_dex,
            "token_pair": opportunity.token_pair,
            "size_sol": float(self.max_trade_size_sol),
            "expected_profit": float(opportunity.profit_after_fees),
            "executed": False,
            "status": "simulated"
        }
        
        self.trade_history.append(trade_record)
        
        # Update P&L tracking
        self.total_pnl += opportunity.profit_after_fees
        
        return True
    
    async def run_monitoring_cycle(self):
        """Run one monitoring cycle"""
        logger.info("Starting monitoring cycle...")
        
        # Find opportunities
        opportunities = await self.find_arbitrage_opportunities()
        
        if opportunities:
            best = opportunities[0]
            logger.info(f"Best opportunity: {best.spread_percent:.2f}% spread, "
                       f"{best.profit_after_fees:.4f} USDC profit")
            
            # Execute best opportunity
            await self.execute_arbitrage(best)
        else:
            logger.info("No opportunities found this cycle")
        
        # Log status
        logger.info(f"Total P&L: {self.total_pnl:.4f} USDC")
        logger.info(f"Total fees: {self.total_fees:.4f} USDC")
    
    async def run(self, interval_seconds: int = 10):
        """Main bot loop"""
        logger.info("Starting arbitrage bot...")
        
        await self.connect()
        
        try:
            while True:
                await self.run_monitoring_cycle()
                logger.info(f"Sleeping for {interval_seconds} seconds...")
                await asyncio.sleep(interval_seconds)
                
        except KeyboardInterrupt:
            logger.info("Bot stopped by user")
        finally:
            await self.disconnect()
            self.save_results()
    
    def save_results(self):
        """Save trade history and results"""
        results = {
            "wallet": str(self.wallet_address),
            "total_pnl": float(self.total_pnl),
            "total_fees": float(self.total_fees),
            "trade_count": len(self.trade_history),
            "trades": self.trade_history,
            "timestamp": datetime.now().isoformat()
        }
        
        filename = f"trading_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {filename}")


async def main():
    """Main entry point"""
    # Wallet address from user
    WALLET_ADDRESS = "8312uUi1DfdNDeSLMCdEhGNrMCQkSdFTMHw9oNmHzTXe"
    
    # Initialize bot
    bot = SolanaArbitrageBot(
        wallet_address=WALLET_ADDRESS,
        min_spread_percent=Decimal("0.5"),  # 0.5% minimum spread
        max_trade_size_sol=Decimal("0.01")  # 0.01 SOL per trade (~$2.50)
    )
    
    # Run bot
    await bot.run(interval_seconds=10)


if __name__ == "__main__":
    asyncio.run(main())
