# Trades-Dynamics-(requires-Deutsche-Boerse-A7-access)
Get a feel of trades driving the equity options market

Abstract :
It is often difficult to know what's going on in the Equity Options market.

But it is especially important to detect even small trading patterns because it's easy to miss new information
which would be relevant for options pricing. Trades can reveal that something is going on.

One important aspect of a trade analysis is the interest behind it. Whether the aggressor was the buyer or the seller,
it  doesn't tell us who was actually crossing the spread to make the trade happen (if any). To figure out if the interest
was the buyer or the seller and how aggressive it was, we will first calibrate a volatility surface in order to get
a theoretical bid and offer price, undisturbed by local (ie. strike specific) microstructure action. The aggressivity 
parameter is defined as such :

aggressivity = min(1, max(-1, (traded_price - mid_theo) / half_theo_spread))

NB : aggressivity is negative for selling interest and positive for buying ones.

This theoretical price will then be used in conjunction with the vega of the trade to determine the intensity of a
a trade. It is defined as :

intensity = vega * aggressivity

This metrics among others will then be used to identify clusters of similar trades. These clusters will in turn be sorted
by vega_aggressivity in order to be able to report on the most remarkable trade action in the period.


How to proceed :
You must first get a Deutsche Boerse A7 subscription in ortder to access intraday data.
Once you have your API key, you can run the Trades Dynamics Jupyter Notebook.

You will need a Python 3.8 interpreter with the following packages :
- QuantLib
- numpy, pandas
- math, datetime, matplotlib
- sklearn, scipy
- requests, warnings

You will a need to download the ad hoc classes provided in this git :
DateAndTime, PricingAnd Calibration, Clustering, TradeFlesh



