# -----------
# - CONTENT -
# -----------

# Imports
# Iterables
# Numbers

# ------------------------------

# -----------
# - IMPORTS -
# -----------
import pandas as pd

# -------------
# - ITERABLES -
# -------------

def cast_to_iterable(value):
    """Return a ``list`` if the input object is not a ``list`` or ``tuple``."""
    if isinstance(value, (list, tuple)):
        return value

    return [value]


# ------------
# - DATETIME -
# ------------

def convert_to_timedelta(column):
    """Convert a ``pandas.Series`` to one with dtype ``timedelta``.

    ``pd.to_timedelta`` does not handle nans, so this function masks the nans, converts and then
    reinserts them.

    Args:
        column (pandas.Series):
            Column to convert.

    Returns:
        pandas.Series:
            The column converted to timedeltas.
    """
    nan_mask = pd.isna(column)
    column[nan_mask] = 0
    column = pd.to_timedelta(column)
    column[nan_mask] = pd.NaT
    return column


# -----------
# - NUMBERS -
# -----------

def convert_floats_to_ints(float_array):
    return [int(x) for x in float_array]

