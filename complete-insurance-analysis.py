import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set styling for better visualizations
plt.style.use('seaborn')
sns.set_palette("husl")

def load_and_preprocess_data():
    """
    Load and preprocess the insurance dataset
    """
    # Load the dataset
    df = pd.read_csv('https://raw.githubusercontent.com/IBM/employee-attrition-aif360/master/data/insurance.csv')
    
    # Create a copy to preserve original data
    data = df.copy()
    
    # Handle categorical variables
    le = LabelEncoder()
    data['sex'] = le.fit_transform(data['sex'])
    data['smoker'] = le.fit_transform(data['smoker'])
    data['region'] = le.fit_transform(data['region'])
    
    return data

def analyze_basic_stats(df):
    """
    Analyze basic statistics and print insights
    """
    print("\n=== Basic Statistics ===")
    print(f"Total number of insurance records: {len(df)}")
    print("\nAverage insurance charges by region:")
    print(df.groupby('region')['charges'].mean().round(2))
    print("\nAverage insurance charges by smoking status:")
    print(df.groupby('smoker')['charges'].mean().round(2))
    
    return df.describe()

def create_visualizations(df):
    """
    Create comprehensive visualizations for analysis
    """
    # Create a figure with multiple subplots
    plt.figure(figsize=(20, 12))
    
    # 1. Distribution of Insurance Charges
    plt.subplot(2, 3, 1)
    sns.histplot(df['charges'], bins=30, kde=True)
    plt.title('Distribution of Insurance Charges')
    plt.xlabel('Charges ($)')
    
    # 2. Age vs Charges
    plt.subplot(2, 3, 2)
    sns.scatterplot(data=df, x='age', y='charges', hue='smoker')
    plt.title('Age vs Insurance Charges')
    
    # 3. BMI vs Charges
    plt.subplot(2, 3, 3)
    sns.scatterplot(data=df, x='bmi', y='charges', hue='smoker')
    plt.title('BMI vs Insurance Charges')
    
    # 4. Average Charges by Region
    plt.subplot(2, 3, 4)
    sns.barplot(data=df, x='region', y='charges')
    plt.title('Average Charges by Region')
    
    # 5. Charges by Smoking Status
    plt.subplot(2, 3, 5)
    sns.boxplot(data=df, x='smoker', y='charges')
    plt.title('Charges Distribution by Smoking Status')
    
    # 6. Correlation Heatmap
    plt.subplot(2, 3, 6)
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    
    plt.tight_layout()
    plt.savefig('insurance_analysis.png')
    plt.close()

def train_prediction_model(df):
    """
    Train and evaluate the prediction model
    """
    # Prepare features and target
    X = df[['age', 'sex', 'bmi', 'children', 'smoker', 'region']]
    y = df['charges']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return model, r2, rmse, feature_importance

def generate_insights_report(df, r2, rmse, feature_importance):
    """
    Generate a comprehensive insights report
    """
    report = """
    === Insurance Claims Analysis Report ===
    
    Dataset Overview:
    - Total Records: {total_records}
    - Average Insurance Charge: ${avg_charge:.2f}
    - Minimum Charge: ${min_charge:.2f}
    - Maximum Charge: ${max_charge:.2f}
    
    Model Performance:
    - R² Score: {r2:.3f}
    - RMSE: ${rmse:.2f}
    
    Key Factors Influencing Insurance Charges:
    {factors}
    
    Population Statistics:
    - Average Age: {avg_age:.1f} years
    - Average BMI: {avg_bmi:.1f}
    - Smokers Percentage: {smoker_pct:.1f}%
    
    Regional Distribution:
    {regional_dist}
    """.format(
        total_records=len(df),
        avg_charge=df['charges'].mean(),
        min_charge=df['charges'].min(),
        max_charge=df['charges'].max(),
        r2=r2,
        rmse=rmse,
        factors=feature_importance.to_string(),
        avg_age=df['age'].mean(),
        avg_bmi=df['bmi'].mean(),
        smoker_pct=(df['smoker'].mean() * 100),
        regional_dist=df.groupby('region')['charges'].agg(['count', 'mean']).round(2).to_string()
    )
    
    return report

def main():
    """
    Main function to run the complete analysis
    """
    print("Starting Insurance Claims Analysis...")
    
    # Load and preprocess data
    df = load_and_preprocess_data()
    
    # Basic statistical analysis
    stats = analyze_basic_stats(df)
    
    # Create visualizations
    create_visualizations(df)
    
    # Train and evaluate model
    model, r2, rmse, feature_importance = train_prediction_model(df)
    
    # Generate insights report
    report = generate_insights_report(df, r2, rmse, feature_importance)
    
    # Save report to file
    with open('insurance_analysis_report.txt', 'w') as f:
        f.write(report)
    
    print("\nAnalysis completed successfully!")
    print(f"Model R² Score: {r2:.3f}")
    print(f"Model RMSE: ${rmse:.2f}")
    print("\nFiles generated:")
    print("1. insurance_analysis.png - Visualizations")
    print("2. insurance_analysis_report.txt - Detailed Report")

if __name__ == "__main__":
    main()
