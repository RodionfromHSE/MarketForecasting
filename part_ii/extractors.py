from abc import ABC, abstractmethod, abstractproperty
from typing import Callable
import pandas as pd
from statsmodels.tsa.stattools import coint
import numpy as np
from sklearn.linear_model import LogisticRegression

class FeatureExtractor(ABC):
    """Base class for feature extractors

    Define:
    - get_params: list of tuples with parameters and suffixes
    - _update_df: method to update lob DataFrame
    - _feature_name: method to extract feature called 'name'
    - _post_process: method to post process features (optional)
    """
    WINDOW_SIZE = 1000
    def __init__(self):
        # all methods starting with _feature
        self.feature_methods = [(method.replace('_feature_', ''), getattr(self, method)) 
                    for method in dir(self)
                    if method.startswith('_feature')]
        assert len(self.feature_methods) > 0, 'No feature methods found'
    
    @property
    def get_params(self) -> list[dict]:
        """Return a list of parameters and parameters suffixes"""
        return [(dict(), '')]

    @abstractmethod
    def _update_df(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    # def _feature_name(self, window: pd.DataFrame,
    #                     *args, **kwargs) -> float:
    #     window = update_df.loc[index_window]
    #     return value

    def _post_process(self, features: pd.DataFrame, suffix: str) -> pd.DataFrame:
        return features

    def _get_slice(self, feature_extractor_method: Callable):
        def wrapper(index: np.array, full_df: pd.DataFrame, kwargs: dict):
            slice_df = full_df.loc[index]
            return feature_extractor_method(slice_df, **kwargs)
        return wrapper

    def _extract_features(self, df: pd.DataFrame, features: pd.DataFrame, params: dict, suffix: str) -> None:
        for name, method in self.feature_methods:
            improved_method = self._get_slice(method)
            feature = df.index.to_series() \
                .rolling(window=self.WINDOW_SIZE) \
                .apply(improved_method, raw=True, args=(df, params))
            features[name + suffix] = feature
        features = self._post_process(features, suffix)

    def extract(self, lob: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
        """Extract features from lob and trades
        Both lob and trades should be sorted by timestamp
        """
        updated_lob = self._update_df(lob)
        features = pd.DataFrame()
        for params, suffix in self.get_params:
            self._extract_features(updated_lob, features, params, suffix)
        
        return features.dropna()

class AutocorrelationFeature(FeatureExtractor):
    def _update_df(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[['mid']]
    
    @property
    def get_params(self) -> list[dict]:
        return [(dict(lag=100), '_100'), (dict(lag=200), '_200')]

    def _feature_autocorrelation(self, df_window: pd.DataFrame, lag: int) -> float:
        return df_window.mid.autocorr(lag=lag)

class DonchianChannelsFeature(FeatureExtractor):
    def _update_df(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[['high', 'low']]

    @property
    def get_params(self) -> list[dict]:
        return [(dict(period=100), '_100'), (dict(period=200), '_200')]
    
    def _feature_donchian_high(self, df_window: pd.DataFrame, period: int) -> float:
        return df_window.high[-period:].max()
    
    def _feature_donchian_low(self, df_window: pd.DataFrame, period: int) -> float:
        return df_window.low[-period:].min()
    
    def _post_process(self, features: pd.DataFrame, suffix: str) -> pd.DataFrame:
        high_name, low_name, middle_name = 'donchian_high' + suffix, 'donchian_low' + suffix, 'donchian_average' + suffix
        features[middle_name] = (features[high_name] + features[low_name]) / 2
        return features

class HighestHighFeature(FeatureExtractor):
    def _update_df(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[['high']]
    
    def _feature_highest_high(self, df_window: pd.DataFrame) -> float:
        return df_window.high.max()

class COGFeature(FeatureExtractor):
    def _update_df(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[['mid']]
    
    @property
    def get_params(self) -> list[dict]:
        k_values = [25, 50, 100, 250, 500]
        r_values = [0.5, 0.25, 0.1, 0.05, 0.01]
        return [(dict(k=k, r=r), f'_k{k}_r{r}') for k in k_values for r in r_values]
    
    def _feature_cog(self, df_window: pd.DataFrame, k: int, r: float) -> float:
        M = df_window.mid.iloc[-1]
        M_k = df_window.mid.iloc[-k] if len(df_window) > k else M
        return (M + r * M_k) / (M + M_k)

class HeikinAshiFeature(FeatureExtractor):
    def _update_df(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[['mid', 'high', 'low']]
    
    def _feature_cur_open(self, df_window: pd.DataFrame) -> float:
        return df_window.mid.iloc[-(1 + self.WINDOW_SIZE // 2)]
    
    def _feature_cur_close(self, df_window: pd.DataFrame) -> float:
        return df_window.mid.iloc[-1]

    def _feature_cur_high(self, df_window: pd.DataFrame) -> float:
        return df_window.high.iloc[-1]
    
    def _feature_cur_low(self, df_window: pd.DataFrame) -> float:
        return df_window.low.iloc[-1]
    
    def _feature_prev_open(self, df_window: pd.DataFrame) -> float:
        return df_window.mid.iloc[0]
    
    def _feature_prev_close(self, df_window: pd.DataFrame) -> float:
        return df_window.mid.iloc[self.WINDOW_SIZE // 2]
    
    def _post_process(self, features: pd.DataFrame, suffix: str) -> pd.DataFrame:
        names = ['cur_open', 'cur_close', 'cur_high', 'cur_low', 'prev_open', 'prev_close']
        name_to_col = {name: name + suffix for name in names}
        cur_close, cur_open, cur_high, cur_low, prev_open, prev_close = [features[name_to_col[name]] for name in names]
        
        features['heikin_open' + suffix] = (prev_open + prev_close) / 2
        features['heikin_close' + suffix] = (cur_open + cur_close + cur_high + cur_low) / 4
        features['heikin_high' + suffix] = features[['cur_high', 'cur_open', 'cur_close']].max(axis=1)
        features['heikin_low' + suffix] = features[['cur_low', 'cur_open', 'cur_close']].min(axis=1)

        features.drop([name_to_col[name] for name in names], axis=1, inplace=True)

        return features

class PriceVolumeDerivationFeature(FeatureExtractor):
    LEVELS = 3
    def _update_df(self, df: pd.DataFrame) -> pd.DataFrame:
        price_cols = [f'asks[{i}].price' for i in range(self.LEVELS)] + [f'bids[{i}].price' for i in range(self.LEVELS)]
        volume_cols = [f'asks[{i}].amount' for i in range(self.LEVELS)] + [f'bids[{i}].amount' for i in range(self.LEVELS)]

        price_avg = df[price_cols].mean(axis=1)
        volume_sum = df[volume_cols].sum(axis=1)
        return pd.DataFrame({'price_avg': price_avg, 'volume_sum': volume_sum})

    def _feature_price_der(self, df_window: pd.DataFrame) -> float:
        return df_window.price_avg.diff().mean()

    def _feature_volume_der(self, df_window: pd.DataFrame) -> float:
        return df_window.volume_sum.diff().mean()

class BestLOBLevelFeature(FeatureExtractor):
    def _update_df(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[['bids[0].amount', 'asks[0].amount']]
    
    def _feature_bid_volume(self, df_window: pd.DataFrame) -> float:
        return df_window['bids[0].amount'].sum()
    
    def _feature_ask_volume(self, df_window: pd.DataFrame) -> float:
        return df_window['asks[0].amount'].sum()

class CointegrationBooleanVectorFeature(FeatureExtractor):
    PARTS = 10  # Number of parts to split the window

    def _update_df(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[['mid']]  # Assuming we are using the 'mid' price for cointegration

    def _feature_cointegr_bool(self, df_window: pd.DataFrame) -> int:
        n = len(df_window)
        part_size = n // self.PARTS
        last_part = df_window.mid[-part_size:]  # Last part is the 10th part
        
        cointegration_vector = []
        
        # Loop through the first 9 parts and check cointegration with the last part
        for i in range(self.PARTS - 1):
            part_i = df_window.mid[i * part_size:(i + 1) * part_size]
            
            # Perform cointegration test between part_i and the last part
            score, p_value, _ = coint(part_i, last_part)
            
            # If p_value is small (indicating cointegration), add 1 to the vector, otherwise 0
            cointegration_vector.append(1 if p_value < 0.05 else 0)
        
        # Convert the list to an integer bit vector (e.g., [1, 0, 1, 0] -> 1010 -> 10)
        cointegration_value = int(''.join(map(str, cointegration_vector)), 2)
        
        return cointegration_value

    def post_process(self, features: pd.DataFrame, suffix: str) -> pd.DataFrame:
        name = 'cointegr_bool' + suffix
        for i in range(self.PARTS - 1):
            features[f'cointegration_{i}{suffix}'] = features[name] & (1 << i)
        return features

class PriceVolumeMeans(FeatureExtractor):
    LEVELS = 5
    def _update_df(self, df: pd.DataFrame) -> pd.DataFrame:
        vol_cols = [f'asks[{i}].amount' for i in range(self.LEVELS)] + [f'bids[{i}].amount' for i in range(self.LEVELS)]
        price_cols = [f'asks[{i}].price' for i in range(self.LEVELS)] + [f'bids[{i}].price' for i in range(self.LEVELS)]
        price_avg = df[price_cols].mean(axis=1)
        volume_avg = df[vol_cols].mean(axis=1)
        return pd.DataFrame({'price_avg': price_avg, 'volume_avg': volume_avg})
    
    def _feature_volume_mean(self, df_window: pd.DataFrame) -> float:
        return df_window.volume_avg.mean()

    def _feature_price_mean(self, df_window: pd.DataFrame) -> float:
        return df_window.price_avg.mean()

class SpreadMidPriceFeature(FeatureExtractor):
    def _update_df(self, df: pd.DataFrame) -> pd.DataFrame:
        spread = df['asks[0].price'] - df['bids[0].price']
        return pd.DataFrame({'spread': spread, 'mid': df.mid})
    
    def _feature_avg_spread(self, df_window: pd.DataFrame) -> float:
        return df_window.spread.mean()
    
    def _feature_last_spread(self, df_window: pd.DataFrame) -> float:
        return df_window.spread.iloc[-1]

    def _feature_avg_midprice(self, df_window: pd.DataFrame) -> float:
        return df_window.mid.mean()
    
    def _feature_last_midprice(self, df_window: pd.DataFrame) -> float:
        return df_window.mid.iloc[-1]

class LogRegEstimatorFeature(FeatureExtractor):
    INPUT_SIZE = 200
    OFFSET = 200
    def _update_df(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[['mid']]
    
    def _form_dataset(self, df_window: pd.DataFrame) -> pd.DataFrame:
        X = df_window.mid.shift(self.OFFSET).dropna().values
        X_windows = np.array([X[i:i+self.INPUT_SIZE] for i in range(len(X) - self.INPUT_SIZE)])

        std = X_windows.std(axis=1)
        y = df_window.mid.shift(-(self.OFFSET + self.INPUT_SIZE)).dropna().values
        y_diff = y - X_windows[:, -1]
        target = (np.abs(y_diff) > std / 2).astype(int)

        last_window = df_window.mid.iloc[-self.INPUT_SIZE:].values.reshape(1, -1)
        return X_windows, target, last_window

    def _feature_log_reg(self, df_window: pd.DataFrame) -> float:
        X, y, last_window = self._form_dataset(df_window)
        if len(np.unique(y)) == 1:
            return y[0]
        model = LogisticRegression()
        model.fit(X, y)
        estimate = model.predict(last_window)
        return estimate
