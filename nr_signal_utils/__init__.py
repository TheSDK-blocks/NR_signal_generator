"""
5G NR downlink signal generation and analysis for DSP DUT testing.

generator.generate builds one period of a cyclic multi-carrier signal from a
list of carrier.Carrier, and analyzer.analyze measures the equalized symbols,
bits and EVM of each carrier from one period of the received signal.
"""
