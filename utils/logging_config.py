# utils/logging_config.py
import logging
import warnings
import os
import sys
from contextlib import contextmanager, redirect_stdout, redirect_stderr
import lightgbm as lgb

@contextmanager
def suppress_stdout_stderr():
    """Context manager to suppress all stdout and stderr."""
    with open(os.devnull, 'w') as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            yield

class MLLoggingSuppressor:
    def __init__(self):
        self.original_level = logging.root.getEffectiveLevel()
        
    def __enter__(self):
        # Silence all logging temporarily
        logging.root.setLevel(logging.ERROR)
        
        # Silence specific loggers
        for logger_name in ['prophet', 'cmdstanpy', 'lightgbm', 'stan_model']:
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.ERROR)
            logger.propagate = False
        
        # Silence warnings
        warnings.filterwarnings('ignore')
        
        # LightGBM specific parameters
        self.lgb_params = {
            'verbose': -1,
            'silent': True,
            'quiet': True
        }
        return self.lgb_params
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original logging level
        logging.root.setLevel(self.original_level)

def train_with_suppressed_output(func):
    """Decorator to suppress all output during model training"""
    def wrapper(*args, **kwargs):
        with MLLoggingSuppressor() as lgb_params:
            with suppress_stdout_stderr():
                if 'params' in kwargs:
                    kwargs['params'].update(lgb_params)
                else:
                    kwargs['params'] = lgb_params
                return func(*args, **kwargs)
    return wrapper

def configure_logging(enable_ml_logging=False):
    """Configure logging levels for all components"""
    # Set basic logging configuration
    logging.basicConfig(level=logging.INFO)
    
    if not enable_ml_logging:
        suppressor = MLLoggingSuppressor()
        return suppressor
    return None 