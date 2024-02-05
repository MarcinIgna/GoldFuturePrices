# Gold Future Prices Prediction

## Overview

This project focuses on predicting future gold prices using machine learning models. The dataset used for training the models is sourced from [Kaggle](https://www.kaggle.com/datasets/gvyshnya/gold-future-prices/data), containing historical gold future prices. The prediction for future prices is made by training various regression models, including Linear Regression, Ridge Regression, Lasso Regression, and ElasticNet Regression.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/MarcinIgna/GoldFuturePrices.git
   ```

2. Navigate to the project directory:

   ```bash
   cd GoldFuturePrices
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root and add your API token from [GoldAPI](https://www.goldapi.io/):

   ```
   GOLD_API_TOKEN=your-api-token-here
   ```

## Usage

Run the main script to train models on historical data, make predictions for future dates using the GoldAPI, and evaluate model performance:

```bash
python main.py
```

## Contributing

Feel free to contribute by opening issues, suggesting improvements, or submitting pull requests.