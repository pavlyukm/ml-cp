# Sugar Beet Yield Prediction

A machine learning project that predicts sugar beet yields using farm management data and weather conditions. This project demonstrates how weather patterns, management practices, and agricultural inputs affect crop productivity.

We developed a machine learning pipeline that:

- Integrates farm management data with weather information
- Identifies key factors affecting sugar beet yields
- Provides reliable yield predictions


Usage Example
```pythonfrom utils import run_complete_pipeline

# Load your data
sugar_beet_data = pd.read_excel('data/Цукровий Буряк.xlsx')
weather_data = pd.read_excel('data/Погода.xlsx')

# Run the complete ML pipeline
models, results, importance_df, processed_data = run_complete_pipeline(
    sugar_beet_data, weather_data
)

# View results
print(f"Best model R²: {max(results.values(), key=lambda x: x['test_r2'])['test_r2']:.3f}")
