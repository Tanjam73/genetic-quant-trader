"""
transaction_costs.py
────────────────────
Models realistic transaction costs for backtesting.

Components modeled:
  1. Commission   — broker fee per share or per notional value
  2. Slippage     — market impact / bid-ask spread
  3. Short rebate — cost to borrow the short leg (optional)

All costs expressed in basis points (bps) per side.
1 bps = 0.01% = 0.0001
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TransactionCostModel:
    """
    Simple additive cost model in basis points per side.

    Parameters
    ----------
    commission_bps : float   Broker commission. Default 5 bps/side (~$0.005/share).
    slippage_bps   : float   Estimated market impact / half-spread. Default 5 bps.
    short_rebate_bps : float Annualized borrow cost for short leg. Default 50 bps/yr.
    """

    commission_bps:    float = 5.0
    slippage_bps:      float = 5.0
    short_rebate_bps:  float = 50.0    # annualized

    # ------------------------------------------------------------------ #
    #  Core cost methods                                                   #
    # ------------------------------------------------------------------ #

    def cost_fraction(self, holding_days: int = 0) -> float:
        """
        Total round-trip cost as a fraction of notional.

        Parameters
        ----------
        holding_days : int
            Days the position was held. Used to prorate short rebate.

        Returns
        -------
        float in (0, 1) representing cost as fraction of position value.
        """
        # Round-trip commission (entry + exit, both legs → 4 sides)
        commission = 4 * self.commission_bps / 10_000

        # Round-trip slippage (entry + exit, both legs → 4 sides)
        slippage = 4 * self.slippage_bps / 10_000

        # Short rebate pro-rated for holding period
        short_rebate = self.short_rebate_bps / 10_000 * (holding_days / 252)

        return commission + slippage + short_rebate

    def commission_only(self) -> float:
        """Commission fraction for a single round-trip (both legs)."""
        return 4 * self.commission_bps / 10_000

    def slippage_only(self) -> float:
        """Slippage fraction for a single round-trip (both legs)."""
        return 4 * self.slippage_bps / 10_000

    # ------------------------------------------------------------------ #
    #  Regime-adjusted costs                                               #
    # ------------------------------------------------------------------ #

    def high_vol_multiplier(self, vix_level: float) -> float:
        """
        Scale slippage up during high-volatility regimes.
        VIX > 30 → 2x slippage. VIX > 50 → 4x slippage.
        """
        if vix_level > 50:
            return 4.0
        elif vix_level > 30:
            return 2.0
        return 1.0

    def adjusted_cost_fraction(
        self, holding_days: int = 0, vix_level: float = 15.0
    ) -> float:
        """
        Total cost with regime-adjusted slippage.
        """
        commission = 4 * self.commission_bps / 10_000
        mult = self.high_vol_multiplier(vix_level)
        slippage = 4 * self.slippage_bps / 10_000 * mult
        short_rebate = self.short_rebate_bps / 10_000 * (holding_days / 252)
        return commission + slippage + short_rebate

    def __repr__(self) -> str:
        return (
            f"TransactionCostModel(commission={self.commission_bps}bps, "
            f"slippage={self.slippage_bps}bps, "
            f"short_rebate={self.short_rebate_bps}bps/yr)"
        )
