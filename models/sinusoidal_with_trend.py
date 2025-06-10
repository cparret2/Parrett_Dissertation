# ----------------------------
# Define a sinusoidal + trend function
# ----------------------------
import numpy as np
def sinusoidal_with_trend(t, a, b, c, d, w, k=0):
    t = np.asarray(t, float)
    if k > 0:
        #K = a * k
        K=k #fixed gov_carrying_capacity
        A = (K - a) / a
        r = b / a
        trend = K / (1 + A * np.exp(-r * t))
    else:
        trend = a + b * t
    cycle = c * np.sin(w * t + d)
    return trend + cycle


