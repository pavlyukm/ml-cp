import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import f_regression
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

def clean_excel_errors(data):
    """Clean Excel error values from dataframe"""
    print("Cleaning Excel error values...")
    
    # Common Excel error values
    excel_errors = ['#REF!', '#VALUE!', '#DIV/0!', '#NAME?', '#N/A', '#NULL!', '#NUM!', 'ERROR']
    
    # Count errors before cleaning
    error_count = 0
    for error_val in excel_errors:
        error_count += (data == error_val).sum().sum()
    
    if error_count > 0:
        print(f"Found {error_count} Excel error values")
        
        # Replace Excel errors with NaN
        for error_val in excel_errors:
            data = data.replace(error_val, np.nan)
        
        print("Replaced Excel errors with NaN")
    
    return data

def remove_leakage_features(data):
    """Remove data leakage features"""
    leakage_features = [
        'Відносна врожайність', 'Вихід Цукру з га', 'Вал Цукру', 'Базова врожайність',
        'Прогноз Цукру', 'Відхилення по цукру', 'Вал', 'Вал Відносної врожайності', 
        'NUE', 'Цукристість', 'Вивезення днів', 'Меляса', 'Періоди вивезення',
        'Густота збирання', 'Польова схожість', 'Тривалість активної вегетації', 
        'Період вегетації', 'Дата збирання_year', 'Дата збирання_month', 
        'Дата збирання_day', 'Дата збирання_dayofweek'
    ]
    return data.drop(columns=leakage_features, errors='ignore')

def create_weather_features(weather_data):
    """Create weather features"""
    print("Creating weather features...")
    
    # Basic yearly aggregations
    yearly_agg = weather_data.groupby('YEAR').agg({
        'Precipitation': ['sum', 'mean', 'max', 'std'],
        'Temperature': ['mean', 'min', 'max', 'std'],
        'Temperature_MAX': ['mean', 'max', 'std'],
        'Temperature_MIN': ['mean', 'min', 'std'],
        'Relative humidity': ['mean', 'min', 'max', 'std'],
        'Wind Speed': ['mean', 'max', 'std'],
        'Insolation': ['sum', 'mean', 'std'],
        'Active TEMP': ['sum', 'mean'],
        'Efective rainfall': ['sum', 'mean', 'max']
    }).reset_index()
    
    # Flatten column names
    yearly_agg.columns = ['YEAR'] + [f'weather_{col[0]}_{col[1]}' for col in yearly_agg.columns[1:]]
    
    # Growing season features (April-September)
    growing_season = weather_data[weather_data['MONTH'].isin([4, 5, 6, 7, 8, 9])]
    if not growing_season.empty:
        growing_agg = growing_season.groupby('YEAR').agg({
            'Precipitation': ['sum', 'mean'],
            'Temperature': ['mean'],
            'Temperature_MAX': ['mean', 'max'],
            'Active TEMP': ['sum'],
            'Insolation': ['sum']
        }).reset_index()
        growing_agg.columns = ['YEAR'] + [f'growing_{col[0]}_{col[1]}' for col in growing_agg.columns[1:]]
        yearly_agg = yearly_agg.merge(growing_agg, on='YEAR', how='left')
    
    # Critical period features (May-July)
    critical_period = weather_data[weather_data['MONTH'].isin([5, 6, 7])]
    if not critical_period.empty:
        critical_agg = critical_period.groupby('YEAR').agg({
            'Precipitation': ['sum', 'mean'],
            'Temperature_MAX': ['mean', 'max'],
            'Temperature': ['mean'],
            'Relative humidity': ['mean']
        }).reset_index()
        critical_agg.columns = ['YEAR'] + [f'critical_{col[0]}_{col[1]}' for col in critical_agg.columns[1:]]
        yearly_agg = yearly_agg.merge(critical_agg, on='YEAR', how='left')
    
    print(f"Created {yearly_agg.shape[1] - 1} weather features")
    return yearly_agg

def integrate_weather(sugar_beet_data, weather_data):
    """Integrate weather data"""
    print("=== Weather Data Integration ===")
    weather_features = create_weather_features(weather_data)
    merged_data = sugar_beet_data.merge(weather_features, left_on='Рік', right_on='YEAR', how='left')
    print(f"Merged data shape: {merged_data.shape}")
    return merged_data

def engineer_features(data):
    """Create engineered features"""
    print("=== Feature Engineering ===")
    data_improved = data.copy()
    
    # Remove year features
    year_features = [col for col in data_improved.columns 
                    if any(term in col.lower() for term in ['рік', 'year', 'drought_year', 'wet_year'])]
    data_improved = data_improved.drop(columns=year_features, errors='ignore')
    
    # Weather interactions
    temp_cols = [col for col in data_improved.columns if 'weather_Temperature' in col and 'mean' in col]
    precip_cols = [col for col in data_improved.columns if 'weather_Precipitation' in col and 'sum' in col]
    
    if temp_cols and precip_cols:
        data_improved['weather_temp_precip_interaction'] = (
            data_improved[temp_cols[0]] * data_improved[precip_cols[0]] / 1000
        )
    
    # Precipitation ratios
    if 'weather_Precipitation_sum' in data_improved.columns:
        precip_median = data_improved['weather_Precipitation_sum'].median()
        data_improved['precipitation_ratio'] = data_improved['weather_Precipitation_sum'] / precip_median
        data_improved['drought_indicator'] = (data_improved['weather_Precipitation_sum'] < precip_median * 0.7).astype(int)
    
    # Growing degree days
    if temp_cols:
        data_improved['growing_degree_days'] = np.maximum(0, data_improved[temp_cols[0]] - 5)
    
    # Fertilizer features
    fertilizer_cols = ['N', 'P2O5', 'K2O']
    available_fert = [col for col in fertilizer_cols if col in data_improved.columns]
    
    if len(available_fert) >= 2:
        data_improved['total_fertilizer'] = data_improved[available_fert].sum(axis=1)
        if 'N' in available_fert and 'K2O' in available_fert:
            data_improved['N_K_ratio'] = data_improved['N'] / (data_improved['K2O'] + 1)
    
    # Management features
    mgmt_cols = ['Безводний Аміак', 'Сидерати', 'Органіка']
    available_mgmt = [col for col in mgmt_cols if col in data_improved.columns]
    if available_mgmt:
        data_improved['management_intensity'] = data_improved[available_mgmt].sum(axis=1)
    
    print(f"Feature engineering complete. Shape: {data_improved.shape}")
    return data_improved

def select_features(X, y, n_features=35):
    """Select best features"""
    print(f"=== Selecting {n_features} features ===")
    
    # Handle non-numeric columns
    non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric_cols:
        for col in non_numeric_cols:
            try:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
            except:
                X = X.drop(columns=[col])
    
    X = X.fillna(X.median())
    
    # Random Forest importance
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    rf_scores = rf.feature_importances_
    
    # Correlations
    correlations = X.corrwith(y).abs().fillna(0)
    
    # F-scores
    f_scores, _ = f_regression(X, y)
    f_scores = np.nan_to_num(f_scores) / np.max(np.nan_to_num(f_scores))
    
    # Combined score
    combined_scores = rf_scores * 0.4 + correlations * 0.3 + f_scores * 0.3
    
    # Select top features
    feature_scores = pd.DataFrame({
        'Feature': X.columns,
        'Score': combined_scores
    }).sort_values('Score', ascending=False)
    
    selected_features = feature_scores.head(n_features)['Feature'].tolist()
    
    weather_selected = len([f for f in selected_features 
                           if any(w in f for w in ['weather_', 'growing_', 'critical_'])])
    print(f"Selected {n_features} features, {weather_selected} are weather-related")
    
    return selected_features, feature_scores

def train_models(X_selected, y):
    """Train XGBoost models"""
    print("=== Training Models ===")
    
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.25, random_state=42)
    
    models = {
        'Conservative': xgb.XGBRegressor(
            n_estimators=900, max_depth=3, learning_rate=0.02,
            subsample=0.5, colsample_bytree=0.7, reg_alpha=1.0,
            reg_lambda=5.0, min_child_weight=10, random_state=42
        ),
        'Balanced': xgb.XGBRegressor(
            n_estimators=1500, max_depth=4, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=1.5,
            reg_lambda=3.0, min_child_weight=6, random_state=42
        ),
        'Performance': xgb.XGBRegressor(
            n_estimators=2000, max_depth=5, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.85, reg_alpha=0,
            reg_lambda=1.0, min_child_weight=4, random_state=42
        )
    }
    
    # Train and evaluate
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        test_r2 = r2_score(y_test, y_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        cv_scores = cross_val_score(model, X_selected, y, cv=3, scoring='neg_mean_squared_error')
        cv_rmse = np.sqrt(-cv_scores.mean())
        
        results[name] = {
            'test_r2': test_r2,
            'test_rmse': test_rmse,
            'cv_rmse': cv_rmse,
            'overfitting': cv_rmse - test_rmse
        }
        
        print(f"{name}: R squared={test_r2:.3f}, RMSE={test_rmse:.2f}, Overfitting={cv_rmse - test_rmse:.2f}")
    
    return models, results


def run_complete_pipeline(sugar_beet_data, weather_data):
    """Run the complete pipeline"""
    print("RUNNING COMPLETE PIPELINE")
    print("="*50)
    
    # Step 0: Clean Excel errors
    sugar_beet_data = clean_excel_errors(sugar_beet_data)
    weather_data = clean_excel_errors(weather_data)
    
    initial_count = len(sugar_beet_data)
    sugar_beet_data['Yield'] = pd.to_numeric(sugar_beet_data['Yield'], errors='coerce')
    sugar_beet_data = sugar_beet_data.dropna(subset=['Yield'])
    print(f"Cleaned target: {initial_count} → {len(sugar_beet_data)} samples")
    
    # Step 1: Remove leakage
    data_clean = remove_leakage_features(sugar_beet_data)
    print(f"After removing leakage: {data_clean.shape}")
    
    # Step 2: Add weather
    data_with_weather = integrate_weather(data_clean, weather_data)
    
    # Step 3: Engineer features
    data_improved = engineer_features(data_with_weather)
    
    # Step 4: Prepare for modeling
    X = data_improved.drop(['Yield'], axis=1, errors='ignore')
    y = data_improved['Yield']
    
    # Additional data cleaning for modeling
    print("Final data cleaning for modeling...")
    
    # Handle any remaining non-numeric values in target
    if y.dtype == 'object':
        y = pd.to_numeric(y, errors='coerce')
        y = y.dropna()
        X = X.loc[y.index]  # Keep only rows with valid target
    
    # Convert all object columns to numeric where possible
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # Fill remaining NaN values
    X = X.fillna(X.median())
    
    print(f"Final dataset for modeling: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Scale weather features
    weather_cols = [col for col in X.columns if any(term in col for term in 
                   ['weather_', 'growing_', 'critical_'])]
    if weather_cols:
        scaler = StandardScaler()
        X_scaled = X.copy()
        X_scaled[weather_cols] = scaler.fit_transform(X_scaled[weather_cols])
        print(f"Scaled {len(weather_cols)} weather features")
    else:
        X_scaled = X.copy()
    
    # Step 5: Select features
    selected_features, feature_scores = select_features(X_scaled, y, n_features=35)
    X_selected = X_scaled[selected_features]
    
    # Step 6: Train models
    models, results = train_models(X_selected, y)
    
    # Step 7: Feature importance
    best_name = max(results.keys(), key=lambda x: results[x]['test_r2'])
    best_model = models[best_name]
    
    importance_df = pd.DataFrame({
        'Feature': selected_features,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    return models, results, importance_df, data_improved

def plot_feature_importance(importance_df, top_n=15):
    """Plot feature importance"""
    plt.figure(figsize=(12, 8))
    
    top_features = importance_df.head(top_n)
    colors = ['lightblue' if any(w in feat for w in ['weather_', 'growing_', 'critical_']) 
              else 'lightcoral' for feat in top_features['Feature']]
    
    plt.barh(range(len(top_features)), top_features['Importance'], color=colors)
    plt.yticks(range(len(top_features)), top_features['Feature'])
    plt.xlabel('Importance Score')
    plt.title(f'Feature Importance (Top {top_n})')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

def create_results_summary(results, importance_df):
    """Create results summary"""
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    
    best_name = max(results.keys(), key=lambda x: results[x]['test_r2'])
    best_result = results[best_name]
    
    print(f"Best Model: {best_name}")
    print(f"R squared: {best_result['test_r2']:.3f}")
    print(f"RMSE: {best_result['test_rmse']:.2f}")
    print(f"Overfitting: {best_result['overfitting']:.2f}")
    print(f"\nTOP 10 FEATURES:")
    weather_count = 0
    for i, row in importance_df.head(10).iterrows():
        emoji = "🌤️" if any(w in row['Feature'] for w in ['weather_', 'growing_', 'critical_']) else "🌱"
        if emoji == "🌤️":
            weather_count += 1
        print(f"   {emoji} {row['Feature']}: {row['Importance']:.3f}")
    
    print(f"\n🌤️ Weather features in top 10: {weather_count}")