import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from pandas import DataFrame
from datetime import datetime, timedelta
from freqtrade.strategy import IStrategy, IntParameter
from freqtrade.strategy.parameters import CategoricalParameter
from freqtrade.persistence import Trade
import talib.abstract as ta
import pandas_ta as pta
from freqtrade.strategy.parameters import DecimalParameter


logger = logging.getLogger(__name__)


class RLStrategy(IStrategy):
    """
    Stratégie d'apprentissage par renforcement pour FreqAI.
    
    Cette stratégie utilise un agent RL pour prendre des décisions de trading
    basées sur l'apprentissage automatique des patterns de marché.
    """
    
    # Paramètre requis par FreqAI - nombre maximum de bougies de démarrage
    startup_candle_count: int = 1440
    
    # Paramètres de base optimisés par hyperopt
    minimal_roi = {
        "0": 0.228,
        "325": 0.122,
        "580": 0.065,
        "840": 0
    }
    
    stoploss = -0.051
    timeframe = '1h'
    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    position_adjustment_enable = False
    
    # Paramètres FreqAI
    model_training_parameters = {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 15,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42
    }
    
    # Paramètres d'hyperopt optimisés - Assouplis pour plus de signaux
    buy_rsi = IntParameter(15, 50, default=34, space="buy", optimize=True)
    sell_rsi = IntParameter(50, 85, default=72, space="sell", optimize=True)
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Méthode requise par IStrategy pour calculer les indicateurs.
        Pour FreqAI, cette méthode doit appeler self.freqai.start()
        """
        # Appel requis par FreqAI selon la documentation
        dataframe = self.freqai.start(dataframe, metadata, self)
        
        # DEBUG: Identifier les colonnes avec NaN
        print("Colonnes avec NaN :\n", dataframe.isna().sum())
        
        # Nettoyage des NaN - CRITIQUE pour éviter les erreurs d'entraînement
        dataframe.fillna(0, inplace=True)
        
        # Calculer les indicateurs de base nécessaires pour le fallback
        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi'] = dataframe['rsi'].fillna(50)  # Valeur neutre
        
        # Moving averages
        dataframe['sma_20'] = ta.SMA(dataframe['close'], timeperiod=20)
        dataframe['sma_20'] = dataframe['sma_20'].fillna(dataframe['close'])
        
        # Volume
        dataframe['volume_sma'] = ta.SMA(dataframe['volume'], timeperiod=20)
        dataframe['volume_sma'] = dataframe['volume_sma'].fillna(dataframe['volume'])
        
        # MACD
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd'].fillna(0)
        dataframe['macdsignal'] = macd['macdsignal'].fillna(0)
        dataframe['macdhist'] = macd['macdhist'].fillna(0)
        
        # Nettoyage final pour s'assurer qu'il n'y a aucun NaN
        dataframe.fillna(0, inplace=True)
        
        return dataframe
    
    def feature_engineering_standard(self, dataframe: DataFrame, **kwargs) -> DataFrame:
        """
        Génération des features finales pour l'apprentissage par renforcement.
        Version simplifiée pour éviter les NaN.
        """
        # Features brutes nécessaires pour RL
        dataframe[f"%-raw_close"] = dataframe["close"]
        dataframe[f"%-raw_open"] = dataframe["open"]
        dataframe[f"%-raw_high"] = dataframe["high"]
        dataframe[f"%-raw_low"] = dataframe["low"]
        dataframe[f"%-raw_volume"] = dataframe["volume"]
        
        # Indicateurs techniques simples (éviter les NaN)
        # RSI avec gestion des NaN
        dataframe['%-rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['%-rsi'] = dataframe['%-rsi'].fillna(50)  # Valeur neutre
        
        # MACD avec gestion des NaN
        macd = ta.MACD(dataframe)
        dataframe['%-macd'] = macd['macd'].fillna(0)
        dataframe['%-macdsignal'] = macd['macdsignal'].fillna(0)
        dataframe['%-macdhist'] = macd['macdhist'].fillna(0)
        
        # Moving averages avec gestion des NaN
        dataframe['%-sma_20'] = ta.SMA(dataframe['close'], timeperiod=20)
        dataframe['%-sma_20'] = dataframe['%-sma_20'].fillna(dataframe['close'])
        
        dataframe['%-ema_20'] = ta.EMA(dataframe['close'], timeperiod=20)
        dataframe['%-ema_20'] = dataframe['%-ema_20'].fillna(dataframe['close'])
        
        # Volume avec gestion des NaN
        dataframe['%-volume_sma'] = ta.SMA(dataframe['volume'], timeperiod=20)
        dataframe['%-volume_sma'] = dataframe['%-volume_sma'].fillna(dataframe['volume'])
        dataframe['%-volume_ratio'] = dataframe['volume'] / dataframe['%-volume_sma']
        dataframe['%-volume_ratio'] = dataframe['%-volume_ratio'].fillna(1)
        
        # Price action simple
        dataframe['%-price_change'] = dataframe['close'].pct_change()
        dataframe['%-price_change'] = dataframe['%-price_change'].fillna(0)
        
        # Features de base sans NaN
        dataframe['%-high_low_ratio'] = dataframe['high'] / dataframe['low']
        dataframe['%-high_low_ratio'] = dataframe['%-high_low_ratio'].fillna(1)
        
        # Crossovers simples
        dataframe['%-sma_cross'] = np.where(dataframe['%-sma_20'] > dataframe['%-ema_20'], 1, -1)
        
        # Features temporelles
        dataframe['%-hour'] = dataframe['date'].dt.hour
        dataframe['%-day_of_week'] = dataframe['date'].dt.dayofweek
        
        # Features décalées (limitées pour éviter les NaN)
        for i in range(1, 3):  # Réduit de 5 à 3
            dataframe[f'%-rsi_shift_{i}'] = dataframe['%-rsi'].shift(i).fillna(50)
            dataframe[f'%-price_change_shift_{i}'] = dataframe['%-price_change'].shift(i).fillna(0)
            dataframe[f'%-volume_ratio_shift_{i}'] = dataframe['%-volume_ratio'].shift(i).fillna(1)
        
        # Nettoyage final CRITIQUE - s'assurer qu'il n'y a aucun NaN
        dataframe.fillna(0, inplace=True)
        
        return dataframe
    
    def feature_engineering_expand_all(self, dataframe: DataFrame, period: int, **kwargs) -> DataFrame:
        """
        Génération de features supplémentaires pour différents timeframes.
        Cette fonction s'étend automatiquement sur indicator_periods_candles, include_timeframes, 
        include_shifted_candles, et include_corr_pairs.
        
        Toutes les features doivent être préfixées par % pour être reconnues par FreqAI.
        """
        # Indicateurs techniques de base
        dataframe[f"%-rsi-period_{period}"] = ta.RSI(dataframe, timeperiod=period)
        dataframe[f"%-rsi-period_{period}"] = dataframe[f"%-rsi-period_{period}"].fillna(50)
        
        dataframe[f"%-mfi-period_{period}"] = ta.MFI(dataframe, timeperiod=period)
        dataframe[f"%-mfi-period_{period}"] = dataframe[f"%-mfi-period_{period}"].fillna(50)
        
        dataframe[f"%-adx-period_{period}"] = ta.ADX(dataframe, timeperiod=period)
        dataframe[f"%-adx-period_{period}"] = dataframe[f"%-adx-period_{period}"].fillna(25)
        
        dataframe[f"%-sma-period_{period}"] = ta.SMA(dataframe['close'], timeperiod=period)
        dataframe[f"%-sma-period_{period}"] = dataframe[f"%-sma-period_{period}"].fillna(dataframe['close'])
        
        dataframe[f"%-ema-period_{period}"] = ta.EMA(dataframe['close'], timeperiod=period)
        dataframe[f"%-ema-period_{period}"] = dataframe[f"%-ema-period_{period}"].fillna(dataframe['close'])
        
        # Bollinger Bands
        bollinger = ta.BBANDS(dataframe, timeperiod=period, nbdevup=2.2, nbdevdn=2.2, matype=0)
        dataframe[f"%-bb_lowerband-period_{period}"] = bollinger['lowerband'].fillna(dataframe['close'])
        dataframe[f"%-bb_middleband-period_{period}"] = bollinger['middleband'].fillna(dataframe['close'])
        dataframe[f"%-bb_upperband-period_{period}"] = bollinger['upperband'].fillna(dataframe['close'])
        
        # Largeur et position relative des Bollinger Bands
        dataframe[f"%-bb_width-period_{period}"] = (
            dataframe[f"%-bb_upperband-period_{period}"] - 
            dataframe[f"%-bb_lowerband-period_{period}"]
        ) / dataframe[f"%-bb_middleband-period_{period}"]
        dataframe[f"%-bb_width-period_{period}"] = dataframe[f"%-bb_width-period_{period}"].fillna(0.1)
        
        dataframe[f"%-close-bb_lower-period_{period}"] = (
            dataframe['close'] / dataframe[f"%-bb_lowerband-period_{period}"]
        )
        dataframe[f"%-close-bb_lower-period_{period}"] = dataframe[f"%-close-bb_lower-period_{period}"].fillna(1)
        
        # Rate of Change
        dataframe[f"%-roc-period_{period}"] = ta.ROC(dataframe['close'], timeperiod=period)
        dataframe[f"%-roc-period_{period}"] = dataframe[f"%-roc-period_{period}"].fillna(0)
        
        # Stochastic
        stoch = ta.STOCH(dataframe, fastk_period=period, slowk_period=3, slowd_period=3)
        dataframe[f"%-slowk-period_{period}"] = stoch['slowk'].fillna(50)
        dataframe[f"%-slowd-period_{period}"] = stoch['slowd'].fillna(50)
        
        # Williams %R
        dataframe[f"%-willr-period_{period}"] = ta.WILLR(dataframe, timeperiod=period)
        dataframe[f"%-willr-period_{period}"] = dataframe[f"%-willr-period_{period}"].fillna(-50)
        
        # Commodity Channel Index
        dataframe[f"%-cci-period_{period}"] = ta.CCI(dataframe, timeperiod=period)
        dataframe[f"%-cci-period_{period}"] = dataframe[f"%-cci-period_{period}"].fillna(0)
        
        # Nettoyage final des NaN
        dataframe.fillna(0, inplace=True)
        
        return dataframe
    
    def feature_engineering_expand_basic(self, dataframe: DataFrame, **kwargs) -> DataFrame:
        """
        Features de base pour l'expansion.
        Cette fonction s'étend sur include_timeframes, include_shifted_candles, et include_corr_pairs.
        """
        # Features de prix
        dataframe['%-pct-change'] = dataframe['close'].pct_change()
        dataframe['%-pct-change'] = dataframe['%-pct-change'].fillna(0)
        
        dataframe['%-pct-change-abs'] = abs(dataframe['%-pct-change'])
        dataframe['%-pct-change-abs'] = dataframe['%-pct-change-abs'].fillna(0)
        
        dataframe['%-high-low-ratio'] = dataframe['high'] / dataframe['low']
        dataframe['%-high-low-ratio'] = dataframe['%-high-low-ratio'].fillna(1)
        
        dataframe['%-close-open-ratio'] = dataframe['close'] / dataframe['open']
        dataframe['%-close-open-ratio'] = dataframe['%-close-open-ratio'].fillna(1)
        
        # Features de volume
        dataframe['%-volume-mean'] = dataframe['volume'].rolling(20).mean()
        dataframe['%-volume-mean'] = dataframe['%-volume-mean'].fillna(dataframe['volume'])
        
        dataframe['%-volume-ratio'] = dataframe['volume'] / dataframe['%-volume-mean']
        dataframe['%-volume-ratio'] = dataframe['%-volume-ratio'].fillna(1)
        
        dataframe['%-volume-pct-change'] = dataframe['volume'].pct_change()
        dataframe['%-volume-pct-change'] = dataframe['%-volume-pct-change'].fillna(0)
        
        # Features de prix brutes (nécessaires pour RL)
        dataframe['%-raw_close'] = dataframe['close']
        dataframe['%-raw_open'] = dataframe['open']
        dataframe['%-raw_high'] = dataframe['high']
        dataframe['%-raw_low'] = dataframe['low']
        dataframe['%-raw_volume'] = dataframe['volume']
        
        # Features temporelles
        dataframe['%-hour'] = dataframe['date'].dt.hour
        dataframe['%-day_of_week'] = dataframe['date'].dt.dayofweek
        dataframe['%-day_of_month'] = dataframe['date'].dt.day
        
        # ATR pour la volatilité
        dataframe['%-atr'] = ta.ATR(dataframe, timeperiod=14)
        dataframe['%-atr'] = dataframe['%-atr'].fillna(dataframe['close'] * 0.02)  # 2% du prix
        
        # Nettoyage final des NaN
        dataframe.fillna(0, inplace=True)
        
        return dataframe
    
    def set_freqai_targets(self, dataframe: DataFrame, **kwargs) -> DataFrame:
        """
        Définition des cibles pour l'apprentissage par renforcement.
        Pour RL, il n'y a pas de cibles directes à définir. C'est un remplissage (neutre)
        jusqu'à ce que l'agent envoie une action.
        
        Selon la documentation officielle FreqAI RL, cette fonction doit définir
        la colonne &-action avec une valeur neutre (0).
        """
        # Pour RL, il n'y a pas de cibles directes à définir
        # Cette valeur sera remplacée par les actions de l'agent
        # La valeur 0 correspond à l'action neutre dans Base5ActionRLEnv
        dataframe["&-action"] = 0
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Logique d'entrée basée sur les actions de l'agent RL.
        Selon la documentation officielle FreqAI RL, l'agent décide des entrées
        via la colonne &-action générée par FreqAI.
        """
        conditions = []
        
        # L'agent RL décide des entrées via la colonne &-action
        # Actions possibles selon Base5ActionRLEnv :
        # 0 = Actions.Neutral (neutre)
        # 1 = Actions.Long_enter (entrée long)
        # 2 = Actions.Short_enter (entrée short)
        # 3 = Actions.Long_exit (sortie long)
        # 4 = Actions.Short_exit (sortie short)
        
        dataframe.loc[:, 'enter_long'] = 0
        dataframe.loc[:, 'enter_short'] = 0
        
        # Vérifier si la colonne &-action existe (générée par FreqAI)
        if '&-action' in dataframe.columns:
            # Utiliser les actions de l'agent RL
            dataframe.loc[dataframe['&-action'] == 1, 'enter_long'] = 1
            dataframe.loc[dataframe['&-action'] == 2, 'enter_short'] = 1
            
            # Log pour debug
            long_signals = (dataframe['enter_long'] == 1).sum()
            short_signals = (dataframe['enter_short'] == 1).sum()
            if long_signals > 0 or short_signals > 0:
                logger.info(f"Agent RL - Signaux long: {long_signals}, Signaux short: {short_signals}")
                
        else:
            # Si la colonne n'existe pas, utiliser des signaux de base
            logger.info("Colonne &-action non trouvée, utilisation de signaux de base")
            
            # Signaux de base plus permissifs basés sur RSI et MACD
            long_condition = (
                (dataframe['rsi'] < 40) &  # RSI plus permissif (30 -> 40)
                (dataframe['volume'] > 0) &
                (dataframe['close'] > dataframe['sma_20']) &
                (dataframe['macd'] > dataframe['macdsignal'])  # MACD bullish
            )
            
            short_condition = (
                (dataframe['rsi'] > 60) &  # RSI plus permissif (70 -> 60)
                (dataframe['volume'] > 0) &
                (dataframe['close'] < dataframe['sma_20']) &
                (dataframe['macd'] < dataframe['macdsignal'])  # MACD bearish
            )
            
            dataframe.loc[long_condition, 'enter_long'] = 1
            dataframe.loc[short_condition, 'enter_short'] = 1
        
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Logique de sortie basée sur les actions de l'agent RL.
        Selon la documentation officielle FreqAI RL, l'agent décide des sorties
        via la colonne &-action générée par FreqAI.
        """
        dataframe.loc[:, 'exit_long'] = 0
        dataframe.loc[:, 'exit_short'] = 0
        
        # Vérifier si la colonne &-action existe (générée par FreqAI)
        if '&-action' in dataframe.columns:
            # Utiliser les actions de l'agent RL pour les sorties
            dataframe.loc[dataframe['&-action'] == 3, 'exit_long'] = 1
            dataframe.loc[dataframe['&-action'] == 4, 'exit_short'] = 1
            
            # Log pour debug
            long_exits = (dataframe['exit_long'] == 1).sum()
            short_exits = (dataframe['exit_short'] == 1).sum()
            if long_exits > 0 or short_exits > 0:
                logger.info(f"Agent RL - Sorties long: {long_exits}, Sorties short: {short_exits}")
                
        else:
            # Si la colonne n'existe pas, utiliser des signaux de base
            logger.info("Colonne &-action non trouvée pour les sorties, utilisation de signaux de base")
            
            # Signaux de sortie basés sur RSI et MACD
            exit_long_condition = (
                (dataframe['rsi'] > 70) |  # RSI sur-acheté
                (dataframe['macd'] < dataframe['macdsignal']) |  # MACD bearish
                (dataframe['close'] < dataframe['sma_20'])  # Prix sous la moyenne
            )
            
            exit_short_condition = (
                (dataframe['rsi'] < 30) |  # RSI survendu
                (dataframe['macd'] > dataframe['macdsignal']) |  # MACD bullish
                (dataframe['close'] > dataframe['sma_20'])  # Prix au-dessus de la moyenne
            )
            
            dataframe.loc[exit_long_condition, 'exit_long'] = 1
            dataframe.loc[exit_short_condition, 'exit_short'] = 1
        
        return dataframe
    
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                       current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Stoploss personnalisé simplifié.
        """
        # Utiliser le stoploss fixe pour éviter les erreurs
        return self.stoploss
    
    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                           time_in_force: str, current_time: datetime, entry_tag: Optional[str],
                           side: str, **kwargs) -> bool:
        """
        Confirmation supplémentaire avant l'entrée en trade.
        """
        # Vérifications supplémentaires si nécessaire
        return True 