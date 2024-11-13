# Modular Stock Analysis App - Technical Architecture

## Core Architecture

### Application Structure
- Entry Point: `app.py` - Dash application with three main tabs (Research, Portfolio, Backtesting)
- Application State: Session-based with a default user_id
- Database: SQLite with thread-safe operations

### Key Components

1. **Data Services Layer**
```
services/
├── stock_data_service.py   # Stock data fetching & caching
├── analysis_service.py     # Technical analysis & backtesting
└── database.py            # SQLite operations
```

- `StockDataService`: 
  - Handles data fetching from yfinance with fallback to alternative sources
  - Implements intelligent caching with SQLite
  - Uses 'Adj Close' prices for all calculations to handle splits/dividends

- `AnalysisService`:
  - Calculates technical indicators and price targets
  - Implements portfolio backtesting logic
  - Generates trading recommendations

- `Database`:
  - Thread-safe SQLite operations
  - Connection pooling and context management
  - Schema includes: users, portfolios, portfolio_holdings

2. **UI Layer**
```
layouts/
├── dashboard_layout.py     # Research dashboard
├── portfolio_layout.py     # Portfolio management
└── backtesting_layout.py  # Strategy testing
```

- Component-based architecture using Dash
- Responsive design with CSS Grid/Flexbox
- Real-time data updates through callbacks

3. **Callback Structure**
```
callbacks/
├── dashboard_callbacks.py    # Stock analysis
├── portfolio_callbacks.py    # Portfolio management
└── backtesting_callbacks.py # Strategy testing
```

- Event-driven architecture
- State management through dcc.Store
- Error handling and user feedback

### Data Flow

1. **Stock Analysis Pipeline**
```
User Input -> StockDataService -> Cache Check -> Data Fetch -> 
Technical Analysis -> Visualization -> UI Update
```

2. **Portfolio Management**
```
User Actions -> Portfolio Model -> Database Operations -> 
UI Updates -> Real-time Analytics
```

3. **Backtesting Flow**
```
Portfolio Selection -> Date Range -> Data Fetching -> 
Performance Calculation -> Results Visualization
```

### Technical Indicators Implementation

```python
class TechnicalIndicators:
    """
    Core technical analysis functionality
    - All calculations use Adj Close prices
    - Rolling windows for indicators
    - Risk metrics calculation
    """
    
    # Key Methods:
    - calculate_moving_averages()
    - calculate_bollinger_bands()
    - calculate_risk_metrics()
    - calculate_all_indicators()
```

### Performance Optimizations

1. **Data Caching**
- SQLite-based cache for stock data
- Partial updates for new data
- Cache invalidation strategy

2. **Computation**
- Vectorized operations with pandas
- Efficient date range handling
- Minimized database operations

3. **UI Performance**
- Lazy loading of components
- Efficient callback patterns
- Client-side callbacks where appropriate

### Key Dependencies
```python
# Core
dash==2.14.2
pandas==2.1.4
numpy==1.26.2

# Data Sources
yfinance==0.2.33
pandas-datareader==0.10.0

# Analysis
scipy==1.11.4
plotly==5.18.0

# Server
Flask==3.0.0
gunicorn==21.2.0
```

### Error Handling Strategy

1. **Data Layer**
- Retries for API calls
- Fallback data sources
- Comprehensive logging

2. **Analysis Layer**
- Input validation
- Numerical computation safety
- Exception propagation

3. **UI Layer**
- User feedback
- Graceful degradation
- Loading states

### Best Practices

1. **Code Organization**
- Clear module boundaries
- Service-oriented architecture
- Consistent naming conventions

2. **Data Handling**
- Always use Adj Close for calculations
- Proper date range management
- Type checking and validation

3. **Performance**
- Efficient data structures
- Minimized recalculations
- Smart caching strategies

### Usage Notes

1. **Development**
```bash
python app.py  # Runs in debug mode
```

2. **Production**
```bash
gunicorn app:server
```

3. **Testing**
```bash
pytest tests/
```

### Extension Points

1. **New Features**
- Additional technical indicators
- More portfolio strategies
- Enhanced backtesting capabilities

2. **Integration**
- Additional data sources
- External APIs
- Authentication systems

3. **Customization**
- Custom indicators
- Portfolio optimization
- Risk management strategies

### Critical Implementation Notes

1. **Price Data**
- Always use 'Adj Close' for calculations
- Handle missing data appropriately
- Validate data quality

2. **Performance**
- Cache management is critical
- Monitor memory usage
- Optimize heavy calculations

3. **Error Handling**
- Comprehensive logging
- User-friendly error messages
- Fallback strategies