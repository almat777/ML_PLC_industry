

#!/usr/bin/env python
# -*- coding: utf-8 -*-


# #  Iron Ore Flotation Process Optimization
# ## Machine Learning for Waste Minimization in Industrial Process
# 
# ---
# 
# ### 📋 Project Description
# 
# **Objective:** Predict silica percentage (Silica %) in final concentrate to minimize waste and optimize iron ore production.
# 
# **Business Task:**
# - Waste minimization (% Silica) → cost reduction
# - Flotation process parameter optimization
# - Quality control automation
# 
# 
# 
# **Dataset:** [Quality Prediction in a Mining Process (Kaggle)](https://www.kaggle.com/datasets/edumagalhaes/quality-prediction-in-a-mining-process)
# 


# ##  1. Import Libraries


# Базовые библиотеки
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Визуализация
import matplotlib.pyplot as plt
import seaborn as sns

# Настройка стиля графиков
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# XGBoost
import xgboost as xgb

# Оптимизация
from scipy.optimize import minimize

print(" Все библиотеки успешно загружены!")
print(f"Pandas version: {pd.__version__}")
print(f"NumPy version: {np.__version__}")

# ##  2. Data Loading
# 
# **Instructions:**
# 1. Download dataset from Kaggle: https://www.kaggle.com/datasets/edumagalhaes/quality-prediction-in-a-mining-process
# 2. Unzip file `MiningProcess_Flotation_Plant_Database.csv`
# 3. Place it in the same folder as this notebook
# 
# Or use Kaggle API:
# ```bash
# kaggle datasets download -d edumagalhaes/quality-prediction-in-a-mining-process
# ```


# Загрузка датасета
df_full = pd.read_csv(r'C:\Users\Almat\Desktop\Machine_Learning\MiningProcess_Flotation_Plant_Database.csv', decimal=',')

print(f"Full dataset uploaded: {df_full.shape[0]:,} строк × {df_full.shape[1]} столбцов")
print(f"Memoty in storage: {df_full.memory_usage(deep=True).sum() / 1024**2:.2f} MB")



df_full.head()

# info about dataset
df_full.info()

# ## 3. Feature Selection and Dataset Simplification
# 
# For optimization, we select 9 key process parameters that have the most impact on silica content:
# 
# **1. Ore Quality (2 features):**
# - `% Iron Feed` - iron content in feed ore
# - `% Silica Feed` - silica content in feed ore
# 
# **2. Reagents (2 features):**
# - `Starch Flow` - starch dosage (depressant)
# - `Amina Flow` - amine dosage (collector)
# 
# **3. Process Conditions (3 features):**
# - `Ore Pulp Flow` - pulp flow rate
# - `Ore Pulp pH` - pulp pH level
# - `Ore Pulp Density` - pulp density
# 
# **4. Flotation Air (2 features):**
# - `Flotation Column 01 Air Flow` - air flow in column 1
# - `Flotation Column 02 Air Flow` - air flow in column 2
# 
# **Decision:** Keep `% Iron Feed` as it provides critical context for ore quality, even though it's not controllable.


# Список выбранных признаков
selected_features = [
    '% Iron Feed',
    '% Silica Feed',
    'Starch Flow',
    'Amina Flow',
    'Ore Pulp Flow',
    'Ore Pulp pH',
    'Ore Pulp Density',
    'Flotation Column 01 Air Flow',
    'Flotation Column 01 Level',
    'Flotation Column 02 Air Flow',
    '% Iron Concentrate',
    '% Silica Concentrate'  # TARGET
]

# Создаем упрощенный датасет
df = df_full[selected_features].copy()

print(f"Simplified dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")

# Сэмплируем 100,000 строк для ускорения обучения
np.random.seed(42)
df = df.sample(n=100000, random_state=42).reset_index(drop=True)

print(f"\n Final dataset: {df.shape[0]:,} строк × {df.shape[1]} столбцов")
print(f" Memory in storage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")


df.head(10)

# ##  4. Exploratory Data Analysis (EDA)
# 
# ### 4.1 Базовая статистика



df.describe()

# ### 4.2 Анализ пропущенных значений


# ==================== FIGURE 1: Missing values ====================

missing_data = pd.DataFrame({
    'Missing_Count': df.isnull().sum(),
    'Missing_Percent': (df.isnull().sum() / len(df)) * 100
}).sort_values('Missing_Percent', ascending=False)

print("\n📋 Missing values by feature:")
print(missing_data[missing_data['Missing_Count'] > 0])

# Visualization of missing values
fig1 = plt.figure(figsize=(12, 6), num='Fig1_Missing_Values')
sns.heatmap(df.isnull(), cbar=True, cmap='viridis', yticklabels=False)
plt.title('Figure 1: Missing Values Heatmap', fontsize=14, fontweight='bold')
plt.xlabel('Features')
plt.tight_layout()

plt.show()      # показываем окно
plt.close(fig1) # по желанию: закрываем фигуру после показа

# ### 4.3 Correlation Analysis


# Вычисляем корреляцию (игнорируя NaN)
correlation_matrix = df.corr()

# Визуализация
# ==================== FIGURE 2: Correlation Matrix ====================

fig2 = plt.figure(figsize=(14, 10), num='Fig2_Correlation_Matrix')

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt='.2f',
    cmap='coolwarm',
    center=0,
    square=True,
    linewidths=1,
    cbar_kws={"shrink": 0.8}
)

plt.title('Figure 2: Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.show()   # только показываем, без сохранения


# Корреляция с целевой переменной (% Silica Concentrate)
target_correlation = correlation_matrix['% Silica Concentrate'].sort_values(ascending=False)

fig3 = plt.figure(figsize=(10, 6))
target_correlation[:-1].plot(kind='barh', color='steelblue')
plt.title('Correlation of features with % Silica Concentrate', fontsize=14, fontweight='bold')
plt.xlabel('Coefficient of correlation' )
plt.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()

print("\nКорреляция с целевой переменной (% Silica Concentrate):")
print(target_correlation)

# ### 4.4 Распределение целевой переменной


# Распределение % Silica Concentrate
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Гистограмма
axes[0].hist(df['% Silica Concentrate'].dropna(), bins=50, color='coral', edgecolor='black', alpha=0.7)
axes[0].set_title('Распределение % Silica Concentrate', fontsize=12, fontweight='bold')
axes[0].set_xlabel('% Silica Concentrate')
axes[0].set_ylabel('Частота')
axes[0].axvline(df['% Silica Concentrate'].mean(), color='red', linestyle='--', 
                linewidth=2, label=f'Среднее: {df["% Silica Concentrate"].mean():.2f}')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Box plot
axes[1].boxplot(df['% Silica Concentrate'].dropna(), vert=True, patch_artist=True,
                boxprops=dict(facecolor='lightblue', color='blue'),
                medianprops=dict(color='red', linewidth=2))
axes[1].set_title('Box Plot % Silica Concentrate', fontsize=12, fontweight='bold')
axes[1].set_ylabel('% Silica Concentrate')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

print(f"Статистика % Silica Concentrate:")
print(f"  Среднее: {df['% Silica Concentrate'].mean():.4f}")
print(f"  Медиана: {df['% Silica Concentrate'].median():.4f}")
print(f"  Станд. отклонение: {df['% Silica Concentrate'].std():.4f}")
print(f"  Минимум: {df['% Silica Concentrate'].min():.4f}")
print(f"  Максимум: {df['% Silica Concentrate'].max():.4f}")

# ### 4.5 Обнаружение выбросов (Outliers)


# Box plots для всех числовых признаков
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
axes = axes.ravel()

for idx, column in enumerate(df.columns):
    axes[idx].boxplot(df[column].dropna(), vert=True, patch_artist=True,
                     boxprops=dict(facecolor='lightgreen', alpha=0.7))
    axes[idx].set_title(column, fontsize=10)
    axes[idx].grid(alpha=0.3)

plt.suptitle('Анализ выбросов (Outliers Detection)', fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.show()

# ##  5. Предобработка данных (Preprocessing)
# 
# ### 5.1 Обработка пропущенных значений


# Стратегия: удаляем строки с пропущенными значениями в целевой переменной
# Остальные - заполняем медианой

print(f"before: {df.shape}")

# Удаляем строки где target = NaN
df = df.dropna(subset=['% Silica Concentrate'])

# Заполняем пропуски медианой для остальных колонок
for col in df.columns:
    if df[col].isnull().sum() > 0:
        median_value = df[col].median()
        df[col].fillna(median_value, inplace=True)
        print(f"  {col}: заполнено {df[col].isnull().sum()} пропусков медианой ({median_value:.2f})")

print(f"\nafter: {df.shape}")
print(f"missings: {df.isnull().sum().sum()}")

# ### 5.2 Feature Engineering
# 
# Создадим новые признаки на основе доменных знаний о процессе флотации.


# 1. Соотношение Iron/Silica на входе
df['Iron_Silica_Ratio'] = df['% Iron Feed'] / (df['% Silica Feed'] + 1e-6)  # +epsilon для избежания деления на 0

# 2. Взаимодействие pH и Density (важно для флотации)
df['pH_Density_Interaction'] = df['Ore Pulp pH'] * df['Ore Pulp Density']

# 3. Расход реагента на единицу потока
df['Starch_per_Flow'] = df['Starch Flow'] / (df['Ore Pulp Flow'] + 1e-6)

print(" Созданы новые признаки:")
print("  - Iron_Silica_Ratio")
print("  - pH_Density_Interaction")
print("  - Starch_per_Flow")
print(f"\nТекущее количество признаков: {df.shape[1]}")

# Проверяем корреляцию новых признаков с target
new_features = ['Iron_Silica_Ratio', 'pH_Density_Interaction', 'Starch_per_Flow']
new_corr = df[new_features + ['% Silica Concentrate']].corr()['% Silica Concentrate'].drop('% Silica Concentrate')

print("Корреляция новых признаков с целевой переменной:")
print(new_corr.sort_values(ascending=False))

# ### 5.3 Разделение на признаки и целевую переменную


# Целевая переменная
target = '% Silica Concentrate'

# Признаки (все кроме target)
X = df.drop(columns=[target])
y = df[target]

print(f"Features (X): {X.shape}")
print(f"Target (y): {y.shape}")
print(f"\nПризнаки для обучения:")
for i, col in enumerate(X.columns, 1):
    print(f"  {i}. {col}")

# ### 5.4 Train-Test Split


# Разделение на train/test (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set: {X_train.shape[0]:,} образцов")
print(f"Test set: {X_test.shape[0]:,} образцов")
print(f"\nСоотношение: {X_train.shape[0]/len(X)*100:.1f}% train / {X_test.shape[0]/len(X)*100:.1f}% test")

# ### 5.5 Масштабирование признаков


# Стандартизация (StandardScaler)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Преобразуем обратно в DataFrame для удобства
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

print(" Признаки масштабированы с помощью StandardScaler")
print(f"\nПример масштабированных данных (первые 5 строк):")
X_train_scaled.head()

# ##  6. Моделирование (Machine Learning)
# 
# Обучим и сравним 4 модели регрессии:
# 1. **Linear Regression** (baseline)
# 2. **Ridge Regression** (регуляризация L2)
# 3. **Random Forest**
# 4. **XGBoost**
# 
# ### 6.1 Функция для оценки моделей


def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """
    Обучает модель и вычисляет метрики качества
    """
    # Обучение
    model.fit(X_train, y_train)
    
    # Предсказания
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Метрики
    metrics = {
        'Model': model_name,
        'Train RMSE': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'Test RMSE': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'Train MAE': mean_absolute_error(y_train, y_train_pred),
        'Test MAE': mean_absolute_error(y_test, y_test_pred),
        'Train R²': r2_score(y_train, y_train_pred),
        'Test R²': r2_score(y_test, y_test_pred)
    }
    
    return metrics, y_test_pred, model

print(" Функция evaluate_model создана")

# ### 6.2 Модель 1: Linear Regression (Baseline)



lr_model = LinearRegression()
lr_metrics, lr_pred, lr_trained = evaluate_model(
    lr_model, X_train_scaled, X_test_scaled, y_train, y_test, 'Linear Regression'
)

print(" Linear Regression Results:")
for key, value in lr_metrics.items():
    if key != 'Model':
        print(f"  {key}: {value:.4f}")

# ### 6.3 Модель 2: Ridge Regression



ridge_model = Ridge(alpha=1.0, random_state=42)
ridge_metrics, ridge_pred, ridge_trained = evaluate_model(
    ridge_model, X_train_scaled, X_test_scaled, y_train, y_test, 'Ridge Regression'
)

print(" Ridge Regression Results:")
for key, value in ridge_metrics.items():
    if key != 'Model':
        print(f"  {key}: {value:.4f}")

# ### 6.4 Модель 3: Random Forest



rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=20,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1  # Использовать все ядра CPU
)

rf_metrics, rf_pred, rf_trained = evaluate_model(
    rf_model, X_train, X_test, y_train, y_test, 'Random Forest'
)

print(" Random Forest Results:")
for key, value in rf_metrics.items():
    if key != 'Model':
        print(f"  {key}: {value:.4f}")

# ### 6.5 Модель 4: XGBoost



xgb_model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1
)

xgb_metrics, xgb_pred, xgb_trained = evaluate_model(
    xgb_model, X_train, X_test, y_train, y_test, 'XGBoost'
)

print(" XGBoost Results:")
for key, value in xgb_metrics.items():
    if key != 'Model':
        print(f"  {key}: {value:.4f}")

# ### 6.6 comparing models



results_df = pd.DataFrame([lr_metrics, ridge_metrics, rf_metrics, xgb_metrics])
results_df = results_df.set_index('Model')



print(results_df.to_string())


# Визуализация сравнения
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

metrics_to_plot = ['Test RMSE', 'Test MAE', 'Test R²']
colors = ['#ff9999', '#66b3ff', '#99ff99']

for idx, metric in enumerate(metrics_to_plot):
    axes[idx].bar(results_df.index, results_df[metric], color=colors[idx], alpha=0.7, edgecolor='black')
    axes[idx].set_title(f'{metric}', fontsize=12, fontweight='bold')
    axes[idx].set_ylabel(metric)
    axes[idx].tick_params(axis='x', rotation=45)
    axes[idx].grid(axis='y', alpha=0.3)
    
    # Добавляем значения на столбцы
    for i, v in enumerate(results_df[metric]):
        axes[idx].text(i, v, f'{v:.4f}', ha='center', va='bottom', fontweight='bold')

plt.suptitle('Сравнение производительности моделей', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

# ### 6.7 Feature Importance (для лучшей модели)


# Выбираем лучшую модель по Test R²
best_model_name = results_df['Test R²'].idxmax()
print(f"Лучшая модель: {best_model_name}")
print(f"   Test R² = {results_df.loc[best_model_name, 'Test R²']:.4f}")

# Для Random Forest или XGBoost получаем feature importance
if best_model_name == 'Random Forest':
    best_model = rf_trained
elif best_model_name == 'XGBoost':
    best_model = xgb_trained
else:
    best_model = None

if best_model is not None:
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    # Визуализация
    plt.figure(figsize=(12, 8))
    plt.barh(feature_importance['Feature'], feature_importance['Importance'], color='teal', alpha=0.7)
    plt.xlabel('Importance', fontsize=12)
    plt.title(f'Feature Importance ({best_model_name})', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("\nТоп-10 важнейших признаков:")
    print(feature_importance.head(10).to_string(index=False))

# ### 6.8 Actual vs Predicted (для лучшей модели)


# Предсказания лучшей модели
if best_model_name == 'Random Forest':
    best_predictions = rf_pred
elif best_model_name == 'XGBoost':
    best_predictions = xgb_pred
elif best_model_name == 'Ridge Regression':
    best_predictions = ridge_pred
else:
    best_predictions = lr_pred

# Scatter plot: Actual vs Predicted
plt.figure(figsize=(10, 6))
plt.scatter(y_test, best_predictions, alpha=0.5, s=10)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         'r--', lw=2, label='Идеальная линия')
plt.xlabel('Фактические значения (Actual)', fontsize=12)
plt.ylabel('Предсказанные значения (Predicted)', fontsize=12)
plt.title(f'Actual vs Predicted - {best_model_name}', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ### 6.9 Residuals Analysis


# Вычисляем остатки (residuals)
residuals = y_test - best_predictions

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Residuals plot
axes[0].scatter(best_predictions, residuals, alpha=0.5, s=10)
axes[0].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[0].set_xlabel('Предсказанные значения', fontsize=11)
axes[0].set_ylabel('Остатки (Residuals)', fontsize=11)
axes[0].set_title('Residuals Plot', fontsize=12, fontweight='bold')
axes[0].grid(alpha=0.3)

# Histogram of residuals
axes[1].hist(residuals, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2)
axes[1].set_xlabel('Остатки (Residuals)', fontsize=11)
axes[1].set_ylabel('Частота', fontsize=11)
axes[1].set_title('Распределение остатков', fontsize=12, fontweight='bold')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

print(f"stat:")
print(f"  mean(avg): {residuals.mean():.6f}")
print(f"  Станд. отклонение: {residuals.std():.4f}")
print(f"  min: {residuals.min():.4f}")
print(f"  max: {residuals.max():.4f}")

# ##  7. Оптимизация параметров процесса
# 
# Используем обученную модель для нахождения оптимальных параметров процесса, которые минимизируют % Silica Concentrate.
# 
# ### 7.1 Определение диапазонов параметров


# Определяем boundaries для оптимизации на основе данных
# Используем min-max из тренировочных данных

# Параметры, которые можно контролировать (исключаем feed и concentrate)
controllable_params = [
    'Starch Flow',
    'Amina Flow', 
    'Ore Pulp Flow',
    'Ore Pulp pH',
    'Ore Pulp Density',
    'Flotation Column 01 Air Flow',
    'Flotation Column 02 Air Flow'
]

# Границы для каждого параметра
bounds = []
for param in controllable_params:
    bounds.append((X_train[param].min(), X_train[param].max()))

print("Диапазоны контролируемых параметров:")
for param, (min_val, max_val) in zip(controllable_params, bounds):
    print(f"  {param:30s}: [{min_val:8.2f}, {max_val:8.2f}]")

# ### 7.2 Оптимизация с помощью scipy.optimize



def objective_function(controllable_values):
    """
    Предсказывает % Silica для заданных контролируемых параметров.
    Неконтролируемые параметры берем как средние из training set.
    """
   
    feature_values = {}
    

    for param, value in zip(controllable_params, controllable_values):
        feature_values[param] = value
   
    uncontrollable = ['% Iron Feed', '% Silica Feed', 'Flotation Column 01 Level', '% Iron Concentrate']
    for param in uncontrollable:
        feature_values[param] = X_train[param].mean()
    
    
    feature_values['Iron_Silica_Ratio'] = feature_values['% Iron Feed'] / (feature_values['% Silica Feed'] + 1e-6)
    feature_values['pH_Density_Interaction'] = feature_values['Ore Pulp pH'] * feature_values['Ore Pulp Density']
    feature_values['Starch_per_Flow'] = feature_values['Starch Flow'] / (feature_values['Ore Pulp Flow'] + 1e-6)
    
    
    X_pred = pd.DataFrame([feature_values])[X_train.columns]
    
   
    predicted_silica = best_model.predict(X_pred)[0]
    
    return predicted_silica




initial_params = [X_train[param].mean() for param in controllable_params]

print("Начальные параметры (средние значения):")
for param, value in zip(controllable_params, initial_params):
    print(f"  {param:30s}: {value:10.2f}")

# Предсказание для начальных параметров
initial_silica = objective_function(initial_params)
print(f"\n% Silica для начальных параметров: {initial_silica:.4f}")

# ОПТИМИЗАЦИЯ
print("Запуск оптимизации...\n")

result = minimize(
    objective_function,
    initial_params,
    method='L-BFGS-B',
    bounds=bounds,
    options={'maxiter': 1000, 'disp': False}
)

if result.success:
    print(" Оптимизация завершена!\n")
    optimal_params = result.x
    optimal_silica = result.fun
else:
    print("Оптимизация не сошлась")
    print(f"Сообщение: {result.message}")

# ### 7.3 Результаты оптимизации



optimization_results = pd.DataFrame({
    'Параметр': controllable_params,
    'Начальное значение': initial_params,
    'Оптимальное значение': optimal_params,
    'Изменение': optimal_params - np.array(initial_params),
    'Изменение (%)': ((optimal_params - np.array(initial_params)) / np.array(initial_params)) * 100
})

print(optimization_results.to_string(index=False))
print("="*100)

print(f"\n ЦЕЛЕВАЯ ПЕРЕМЕННАЯ (% Silica Concentrate):")
print(f"   Начальное значение: {initial_silica:.4f} %")
print(f"   Оптимальное значение: {optimal_silica:.4f} %")
print(f"   Улучшение: {initial_silica - optimal_silica:.4f} % ({((initial_silica - optimal_silica)/initial_silica)*100:.2f}% снижение)")

print(f"\n ЭКОНОМИЧЕСКИЙ ЭФФЕКТ (при производительности 1000 т/ч):")
waste_reduction = (initial_silica - optimal_silica) * 10  # тонн/час
print(f"   Снижение отходов: {waste_reduction:.2f} тонн/час")
print(f"   Экономия в месяц: {waste_reduction * 24 * 30:.0f} тонн")
print(f"   Экономия в год: {waste_reduction * 24 * 365:.0f} тонн")

print("\n" + "="*100)

# ### 7.4 Визуализация результатов оптимизации


# Bar chart сравнения начальных и оптимальных параметров
fig, ax = plt.subplots(figsize=(14, 7))

x = np.arange(len(controllable_params))
width = 0.35

# Нормализуем для визуализации (0-1 scale)
initial_norm = [(initial_params[i] - bounds[i][0]) / (bounds[i][1] - bounds[i][0]) 
                for i in range(len(controllable_params))]
optimal_norm = [(optimal_params[i] - bounds[i][0]) / (bounds[i][1] - bounds[i][0]) 
                for i in range(len(controllable_params))]

ax.bar(x - width/2, initial_norm, width, label='Начальные', alpha=0.8, color='coral')
ax.bar(x + width/2, optimal_norm, width, label='Оптимальные', alpha=0.8, color='lightgreen')

ax.set_xlabel('Параметры процесса', fontsize=12)
ax.set_ylabel('Нормализованное значение (0-1)', fontsize=12)
ax.set_title('Сравнение начальных и оптимальных параметров', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(controllable_params, rotation=45, ha='right')
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

# ### 7.5 Сохранение оптимальных параметров для PLC



import json

plc_setpoints = {}
for param, value in zip(controllable_params, optimal_params):
    # Форматируем имя для PLC (убираем пробелы и спец.символы)
    plc_tag = 'DB_Setpoints.' + param.replace(' ', '_').replace('%', 'Pct')
    plc_setpoints[plc_tag] = float(value)

# Сохраняем в JSON
with open('optimal_setpoints.json', 'w') as f:
    json.dump(plc_setpoints, f, indent=4)

print(" Оптимальные setpoints сохранены в 'optimal_setpoints.json'")
print("\nСодержимое файла:")
print(json.dumps(plc_setpoints, indent=4))

from opcua import Client

url = "opc.tcp://10.103.77.15:4840"
client = Client(url)

def walk(node, text, depth=0, max_depth=6, max_hits=30, hits=None):
    if hits is None:
        hits = []
    if depth > max_depth or len(hits) >= max_hits:
        return hits

    try:
        bn = node.get_browse_name()
        name = f"{bn.NamespaceIndex}:{bn.Name}"
    except:
        name = ""

    if text.lower() in name.lower():
        hits.append((name, node.nodeid.to_string()))

    try:
        for ch in node.get_children():
            walk(ch, text, depth + 1, max_depth, max_hits, hits)
    except:
        pass

    return hits

try:
    client.connect()
    print("Connected")

    ns = client.get_namespace_array()
    print("\nNamespace array:")
    for i, s in enumerate(ns):
        print(i, s)

    objects = client.get_objects_node()

    print("\nSearch for DB_Fall:")
    for name, nid in walk(objects, "DB_Fall", max_depth=8):
        print(name, "->", nid)

    print("\nSearch for Starch_Flow:")
    for name, nid in walk(objects, "Starch_Flow", max_depth=10):
        print(name, "->", nid)

finally:
    client.disconnect()
    print("Disconnected")
