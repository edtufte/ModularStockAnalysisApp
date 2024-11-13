from setuptools import setup, find_packages

setup(
    name="ModularStockAnalysisApp",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'dash==2.14.2',
        'pandas==2.1.4',
        'numpy==1.26.2',
        'yfinance==0.2.33',
        'plotly==5.18.0',
        'scipy==1.11.4',
        'python-dateutil==2.8.2',
        'pytz==2023.3.post1',
        'pandas-datareader==0.10.0',
        'dash-core-components==2.0.0',
        'dash-html-components==2.0.0',
        'dash-table==5.0.0',
        'Flask==3.0.0',
        'gunicorn==21.2.0',
    ],
)