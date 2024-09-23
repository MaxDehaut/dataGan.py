# -----------
# - CONTENT -
# -----------

# Imports
# NumberRandomizer

# ------------------------------

# -----------
# - IMPORTS -
# -----------
import warnings
from enum import Enum

import numpy as np
import pandas as pd
from scipy.stats import gamma

from CastConverter import convert_floats_to_ints


# --------------------
# - DATESETGENERATOR -
# --------------------

class DatasetGenerator:
    """This randomizer ..."""
    
    class FieldType(Enum):
        BOOLEAN = 1
        DATETIME = 2
        NUMBER = 3

    def get_dataframe(self, arr_params, size):
        """Returns a dataset"""
        dtr = DatetimeRandomizer()
        nr = NumberRandomizer()
        df_return = pd.DataFrame()

        for p in arr_params:
            if 'field' in p and 'fieldtype' in p:
                if p['fieldtype'] == DatasetGenerator.FieldType.BOOLEAN:
                    tmp_categories = nr.get_numbers(distribution=NumberRandomizer.Distribution.WEIGHTED,
                                                size=size,
                                                params=p)
                    categories = convert_floats_to_ints(tmp_categories)

                    df_return = pd.concat([df_return, categories], ignore_index=False, axis=1)

                elif p['fieldtype'] == DatasetGenerator.FieldType.DATETIME:
                    timestamps = dtr.get_timestamps_pd(field_name=p['field'], params=p)
                    df_return = pd.concat([df_return, timestamps], ignore_index=False, axis=1)

                elif p['fieldtype'] == DatasetGenerator.FieldType.NUMBER:
                    tmp_numbers = nr.get_numbers(distribution=p['distribution'],
                                                size=size,
                                                params=p)
                    numbers = pd.DataFrame({p['field']: tmp_numbers})
                    df_return = pd.concat([df_return, numbers], ignore_index=False, axis=1)

        return df_return


# ----------------------
# - DATETIMERANDOMIZER -
# ----------------------

class DatetimeRandomizer:
    """This randomizer ..."""

    def get_timestamps(self, field_name, params):
        """Receives timestamp specifications and creates and outputs a pandas df of type timestamp"""
        if 'start' in params:
            return pd.date_range(start=params['start'],
                                end=params['end'],
                                periods=params['periods'],
                                freq=params['freq'],
                                name=field_name).to_pydatetime()
        return None

    def get_timestamps_pd(self, field_name, params):
        """Receives"""
        if 'start' in params:
            return pd.DataFrame(pd.date_range(start=params['start'],
                                                    end=params['end'],
                                                    periods=params['periods'],
                                                    freq=params['freq'],
                                                    name=field_name).to_pydatetime(),
                                                    columns=[field_name])
        return None

# --------------------
# - NUMBERRANDOMIZER -
# --------------------

class NumberRandomizer:
    """This randomizer takes in input for the paramaters depending on the distribution selected.
    It also accepts min and max arguments for the output of the data set in a 'data smapling' 
    method similar to SDV's "reject_sampling" method. Thanks to numpy's extremely efficient array
    creator, the sampling method used barely takes more time."""

    class Distribution(Enum):
        CHISQUARE = 1
        EXPONENTIAL = 2
        GAMMA = 3
        GUMBEL = 4
        LAPLACE = 5
        LOGISTIC = 6
        NONCENTRALCHISQUARE = 7
        NONCENTRALF = 8
        NORMAL = 9
        PARETO = 10
        POISSON = 11
        POWER = 12
        RAYLEIGH = 13
        STDT = 14
        TRIANGULAR = 15
        UNIFORM = 16
        VONMISES = 17
        WALD = 18
        WEIBULL = 19
        WEIGHTED = 20
        ZIPF = 21

    # --------------
    # - RANDOMIZER -
    # --------------
    def get_numbers(self, distribution, size, params):
        """Returns an array of numbers following a specific distribution"""

        new_data = np.array([])
        min = max = None

        if 'min' in params:
            min = params['min']
        if 'max' in params:
            max = params['max']
            # if Min  is greater than max the sampler will always
            # empty the array and this will loop without end
            if min > max:
                warnings.warn('Minimum is higher than maximum!')
                return None

        # Distribution generation loop
        if distribution == NumberRandomizer.Distribution.CHISQUARE:
            if 'df' in params:
                new_data = np.append(new_data, np.random.chisquare(params['df'], size=size))

        elif distribution == NumberRandomizer.Distribution.EXPONENTIAL:
            if 'scale' in params:
                new_data = np.append(new_data, np.random.exponential(params['scale'], size=size))

        elif distribution == NumberRandomizer.Distribution.GAMMA:
            if 'shape' in params:
                new_data = np.append(new_data, gamma.rvs(params['shape'], size=size))

        elif distribution == NumberRandomizer.Distribution.GUMBEL:
            if 'mu' in params and 'beta' in params:
                new_data = np.append(new_data, np.random.gumbel(params['mu'], params['beta'], size=size))

        elif distribution == NumberRandomizer.Distribution.LAPLACE:
            if 'loc' in params and 'scale' in params:
                new_data = np.append(new_data, np.random.laplace(params['loc'], params['scale'], size=size))

        elif distribution == NumberRandomizer.Distribution.LOGISTIC:
            if 'loc' in params and 'scale' in params:
                new_data = np.append(new_data, np.random.logistic(params['loc'], params['scale'], size=size))

        elif distribution == NumberRandomizer.Distribution.NONCENTRALCHISQUARE:
            if 'df' in params and 'nonc' in params:
                new_data = np.append(new_data, np.random.noncentral_chisquare(params['df'],
                                                                            params['nonc'],
                                                                            size=size))

        elif distribution == NumberRandomizer.Distribution.NONCENTRALF:
            if 'dfnum' in params and 'dfden' in params and 'nonc' in params:
                new_data = np.append(new_data, 
                                    np.random.noncentral_f(params['dfnum'],
                                                            params['dfden'],
                                                            params['nonc'],
                                                            size=size))

        elif distribution == NumberRandomizer.Distribution.NORMAL:
            if 'mean' in params and 'std' in params:
                new_data = np.append(new_data, 
                                    np.random.normal(params['mean'], params['std'], size=size))

        elif distribution == NumberRandomizer.Distribution.PARETO:
            if 'shape' in params:
                new_data = np.append(new_data, np.random.pareto(params['shape'], size=size))

        elif distribution == NumberRandomizer.Distribution.POISSON:
            if 'lam' in params:
                new_data = np.append(new_data, np.random.poisson(params['lam'], size=size))

        elif distribution == NumberRandomizer.Distribution.POWER:
            if 'a' in params:
                new_data = np.append(new_data, np.random.power(params['a'], size=size))

        elif distribution == NumberRandomizer.Distribution.RAYLEIGH:
            if 'scale' in params:
                new_data = np.append(new_data, np.random.rayleigh(params['scale'], size=size))

        elif distribution == NumberRandomizer.Distribution.STDT:
            if 'df' in params:
                new_data = np.append(new_data, np.random.standard_t(params['df'], size=size))

        elif distribution == NumberRandomizer.Distribution.TRIANGULAR:
            if 'left' in params and 'mode' in params and 'right' in params:
                new_data = np.append(new_data,
                                    np.random.triangular(params['left'],
                                                        params['mode'],
                                                        params['right'],
                                                        size=size))

        elif distribution == NumberRandomizer.Distribution.UNIFORM:
            if min and max:
                new_data = np.append(new_data, np.random.uniform(min, max, size=size))

        elif distribution == NumberRandomizer.Distribution.VONMISES:
            if 'mu' in params and 'kappa' in params:
                new_data = np.append(new_data, np.random.vonmises(params['mu'], params['kappa'], size=size))

        elif distribution == NumberRandomizer.Distribution.WALD:
            if 'mean' in params and 'scale' in params:
                new_data = np.append(new_data, np.random.wald(params['mean'], params['scale'], size=size))

        elif distribution == NumberRandomizer.Distribution.WEIBULL:
            if 'shape' in params:
                new_data = np.append(new_data, np.random.weibull(params['shape'], size=size))

        elif distribution == NumberRandomizer.Distribution.WEIGHTED:
            if 'a' in params and 'weights' in params:
                new_data = np.append(new_data, 
                                    np.random.choice(params['a'],
                                                    p=params['weights'],
                                                    size=size))

        elif distribution == NumberRandomizer.Distribution.ZIPF:
            if 'a' in params:
                new_data = np.append(new_data, np.random.zipf(params['a'], size=size))


        # Sampler removing values greater than max then smaller than min
        if max != None: 
            new_data = new_data[new_data < max]
        if min != None:
            new_data = new_data[new_data > min]

        return new_data[:size]