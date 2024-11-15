import pandas as pd
import numpy as np
from prophet import Prophet
import lightgbm as lgb
from lightgbm import early_stopping
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import logging
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class MLService:
    def __init__(self):
        self.prophet_model = None
        self.lgb_model = None
        self.feature_importance = None
        self.metrics = None
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features for ML model"""
        df = df.copy()
        
        # Technical indicators as features
        df['returns'] = df['Close'].pct_change()
        df['volatility'] = df['returns'].rolling(window=20).std()
        
        # Price-based features
        df['ma_5'] = df['Close'].rolling(window=5).mean()
        df['ma_20'] = df['Close'].rolling(window=20).mean()
        df['ma_50'] = df['Close'].rolling(window=50).mean()
        
        # Volume features
        df['volume_ma_5'] = df['Volume'].rolling(window=5).mean()
        df['volume_ma_20'] = df['Volume'].rolling(window=20).mean()
        
        # Trend features
        df['price_trend'] = (df['Close'] - df['Close'].shift(5)) / df['Close'].shift(5)
        df['volume_trend'] = (df['Volume'] - df['Volume'].shift(5)) / df['Volume'].shift(5)
        
        # Momentum
        df['momentum'] = df['Close'] - df['Close'].shift(10)
        
        # Clean up NaN values
        df = df.dropna()
        
        return df
        
    def calculate_forward_mape(self, df: pd.DataFrame, forward_days: int = 30) -> float:
        """Calculate MAPE for forward predictions using a rolling window approach"""
        if len(df) < forward_days * 2:
            return None
            
        errors = []
        # Use the last year of data for validation
        validation_period = min(252, len(df) - forward_days)
        step_size = 5  # 5-day steps (weekly)
        
        for i in range(0, validation_period, step_size):
            # Use data up to this point to make a prediction
            train_data = df.iloc[:-(validation_period-i)]
            actual_future = df.iloc[-(validation_period-i)+forward_days]['Close']
            
            # Skip if actual value is too close to zero
            if abs(actual_future) < 1e-6:
                continue
            
            # Prepare data for Prophet
            prophet_df = pd.DataFrame({
                'ds': train_data.index,
                'y': train_data['Close']
            })
            
            # Train model and make prediction
            try:
                model = Prophet(
                    yearly_seasonality=True,
                    weekly_seasonality=True,
                    daily_seasonality=True,
                    changepoint_prior_scale=0.05
                )
                model.fit(prophet_df)
                future = model.make_future_dataframe(periods=forward_days)
                forecast = model.predict(future)
                
                # Get the prediction for the target date
                predicted = forecast.iloc[-1]['yhat']
                
                # Calculate percentage error with capping for extreme values
                error = abs((actual_future - predicted) / actual_future)
                # Cap the error at 100% (1.0) to avoid extreme values
                error = min(error, 1.0)
                errors.append(error)
            except Exception as e:
                logger.warning(f"Error in forward MAPE calculation: {str(e)}")
                continue
        
        if errors:
            # Use median instead of mean to be more robust to outliers
            return np.median(errors) * 100
        return None

    def train_prophet(self, df: pd.DataFrame, prediction_days: int = 30) -> Dict[str, Any]:
        """Train Prophet model for time series prediction"""
        try:
            # Calculate forward-looking MAPE
            forward_mape = self.calculate_forward_mape(df, forward_days=30)
            
            # Prepare data for Prophet
            prophet_df = pd.DataFrame({
                'ds': df.index,
                'y': df['Close']
            })
            
            # Initialize and train Prophet model
            self.prophet_model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=True,
                changepoint_prior_scale=0.05
            )
            self.prophet_model.fit(prophet_df)
            
            # Make future predictions (extending to 60 days)
            future_dates = self.prophet_model.make_future_dataframe(periods=60)
            forecast = self.prophet_model.predict(future_dates)
            
            # Split into historical and future predictions
            historical_mask = forecast['ds'].isin(df.index)
            future_mask = ~historical_mask
            
            # Calculate metrics on training data
            train_predictions = forecast[historical_mask]['yhat']
            mape = mean_absolute_percentage_error(df['Close'], train_predictions) * 100  # Convert to percentage
            rmse = np.sqrt(mean_squared_error(df['Close'], train_predictions))
            
            return {
                'historical_dates': forecast[historical_mask]['ds'].tolist(),
                'historical_values': forecast[historical_mask]['yhat'].tolist(),
                'historical_actual': df['Close'].tolist(),
                'forecast_dates': forecast[future_mask]['ds'].tolist(),
                'forecast_values': forecast[future_mask]['yhat'].tolist(),
                'forecast_lower': forecast[future_mask]['yhat_lower'].tolist(),
                'forecast_upper': forecast[future_mask]['yhat_upper'].tolist(),
                'metrics': {
                    'mape': mape,
                    'forward_mape': forward_mape,  # Add forward-looking MAPE
                    'rmse': rmse
                }
            }
            
        except Exception as e:
            logger.error(f"Error in Prophet training: {str(e)}")
            return None
            
    def train_lgb(self, df: pd.DataFrame, target_days: int = 5) -> Dict[str, Any]:
        """Train LightGBM model for feature importance and additional predictions"""
        try:
            # Prepare features
            feature_df = self.prepare_features(df)
            
            # Create target variable (future returns)
            feature_df['target'] = feature_df['Close'].shift(-target_days) / feature_df['Close'] - 1
            feature_df = feature_df.dropna()
            
            # Select features for training
            feature_cols = [
                'returns', 'volatility', 'ma_5', 'ma_20', 'ma_50',
                'volume_ma_5', 'volume_ma_20', 'price_trend',
                'volume_trend', 'momentum'
            ]
            
            X = feature_df[feature_cols]
            y = feature_df['target']
            
            # Time series split
            tscv = TimeSeriesSplit(n_splits=5)
            
            # Train model
            self.lgb_model = lgb.LGBMRegressor(
                objective='regression',
                n_estimators=100,
                learning_rate=0.05,
                num_leaves=31,
                random_state=42
            )
            
            # Track metrics across folds
            metrics = []
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                self.lgb_model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    eval_metric='rmse',
                    callbacks=[early_stopping(20)]
                )
                
                val_pred = self.lgb_model.predict(X_val)
                fold_mape = mean_absolute_percentage_error(y_val, val_pred) * 100  # Convert to percentage
                fold_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
                metrics.append({'mape': fold_mape, 'rmse': fold_rmse})
            
            # Calculate feature importance
            self.feature_importance = dict(zip(
                feature_cols,
                self.lgb_model.feature_importances_
            ))
            
            # Store average metrics
            self.metrics = {
                'mape': np.mean([m['mape'] for m in metrics]),
                'rmse': np.mean([m['rmse'] for m in metrics])
            }
            
            return {
                'feature_importance': self.feature_importance,
                'metrics': self.metrics
            }
            
        except Exception as e:
            logger.error(f"Error in LightGBM training: {str(e)}")
            return None
    
    def get_predictions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get predictions from trained models"""
        try:
            # Prepare features
            df_features = self.prepare_features(df)
            
            if df_features is None or df_features.empty:
                logger.warning("No features available for prediction")
                return {}
                
            # Get predictions from Prophet model (60 days forecast)
            prophet_results = self.train_prophet(df_features, prediction_days=60)
            if not prophet_results:
                logger.warning("Prophet model failed to generate predictions")
                return {}
                
            # Split historical and future predictions
            historical_mask = pd.to_datetime(prophet_results['forecast_dates']).isin(df.index)
            future_mask = ~historical_mask
            
            forecast_dates = pd.to_datetime(prophet_results['forecast_dates'])
            forecast_values = prophet_results['forecast_values']
            
            # Format predictions for UI
            predictions = {
                'historical_dates': df.index.tolist(),
                'historical_values': np.array(forecast_values)[historical_mask].tolist(),
                'historical_actual': df['Close'].tolist(),
                'forecast_dates': np.array(forecast_dates[future_mask]).astype(str).tolist(),
                'forecast_values': np.array(forecast_values)[future_mask].tolist(),
                'forecast_lower': np.array(prophet_results['forecast_lower'])[future_mask].tolist(),
                'forecast_upper': np.array(prophet_results['forecast_upper'])[future_mask].tolist(),
                'metrics': prophet_results['metrics'],
                'training_period': {
                    'start': df.index[0].strftime('%Y-%m-%d'),
                    'end': df.index[-1].strftime('%Y-%m-%d')
                }
            }
            
            # Get feature importance from LightGBM
            try:
                lgb_results = self.train_lgb(df_features)
                if lgb_results and self.feature_importance:
                    predictions['feature_importance'] = self.feature_importance
            except Exception as e:
                logger.warning(f"LightGBM training failed: {str(e)}")
                # Continue with Prophet predictions even if LightGBM fails
            
            return {'predictions': predictions}
            
        except Exception as e:
            logger.error(f"Error getting predictions: {str(e)}")
            return {}
            
    def predict(self, df: pd.DataFrame, prediction_days: int = 30) -> Dict[str, Any]:
        """Generate predictions using both models"""
        try:
            # Prophet predictions
            prophet_results = self.train_prophet(df, prediction_days)
            
            # LightGBM predictions and feature importance
            lgb_results = self.train_lgb(df)
            
            if prophet_results and lgb_results:
                forecast_df = prophet_results['forecast'].tail(prediction_days)
                
                return {
                    'predictions': {
                        'dates': forecast_df['ds'].tolist(),
                        'values': forecast_df['yhat'].tolist(),
                        'lower_bound': forecast_df['yhat_lower'].tolist(),
                        'upper_bound': forecast_df['yhat_upper'].tolist()
                    },
                    'feature_importance': lgb_results['feature_importance'],
                    'metrics': {
                        'prophet': prophet_results['metrics'],
                        'lightgbm': lgb_results['metrics']
                    }
                }
                
            return None
            
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            return None
            
    def get_market_signals(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate market signals based on ML predictions"""
        try:
            # Get predictions
            predictions = self.predict(df)
            if not predictions:
                return None
                
            # Calculate current trend
            current_price = df['Close'].iloc[-1]
            predicted_prices = predictions['predictions']['values']
            predicted_trend = (predicted_prices[-1] - current_price) / current_price
            
            # Get top features
            top_features = dict(sorted(
                predictions['feature_importance'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:5])
            
            # Generate signal
            if predicted_trend > 0.05:  # 5% threshold
                signal = 'Strong Buy'
                confidence = min(abs(predicted_trend) * 100, 100)
            elif predicted_trend > 0.02:
                signal = 'Buy'
                confidence = min(abs(predicted_trend) * 100, 100)
            elif predicted_trend < -0.05:
                signal = 'Strong Sell'
                confidence = min(abs(predicted_trend) * 100, 100)
            elif predicted_trend < -0.02:
                signal = 'Sell'
                confidence = min(abs(predicted_trend) * 100, 100)
            else:
                signal = 'Hold'
                confidence = 50
                
            return {
                'signal': signal,
                'confidence': confidence,
                'predicted_trend': predicted_trend,
                'top_features': top_features,
                'metrics': predictions['metrics']
            }
            
        except Exception as e:
            logger.error(f"Error in market signals: {str(e)}")
            return None
