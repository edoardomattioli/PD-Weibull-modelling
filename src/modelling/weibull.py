import numpy as np
import pandas as pd

def generate_pd_curve(
    shape: float = 1.5,
    scale: float = 5,
    horizon: int = 20
):

    t = np.arange(1, horizon + 1)

    cumulative_pd = 1 - np.exp(-(t / scale) ** shape)

    df = pd.DataFrame({
        "year": t,
        "cumulative_pd": cumulative_pd
    })

    return df
