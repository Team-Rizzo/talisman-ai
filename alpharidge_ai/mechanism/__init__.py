"""Pure mechanism functions: the economic rules, with no I/O, clock or chain access.

Every validator must compute bit-identical outputs from identical inputs, so nothing
here may read config, the network or the wall clock. Callers pass values in.
"""
