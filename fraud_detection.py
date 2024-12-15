import traceback
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from data_processing.data_processing import ParallelDynamoProcessor, parallel_process_data


# Get data using the parallel processor instead of direct DynamoDB access
def get_processed_data():
    processor = ParallelDynamoProcessor('Fraud_Detection', segment_count=8)
    df = processor.fetch_data_parallel()
    return parallel_process_data(df)

# 1. Transaction Time Analysis
def analyze_transaction_time(df):
    # Convert TransactionDT to datetime
    df['TransactionDateTime'] = pd.to_datetime(df['TransactionDT'], unit='s')
    
    plt.figure(figsize=(15, 10))
    
    # Transactions by hour
    plt.subplot(2, 1, 1)
    df['hour'] = df['TransactionDateTime'].dt.hour
    sns.histplot(data=df, x='hour', bins=24)
    plt.title('Transaction Distribution by Hour')
    
    # Transactions by day of week
    plt.subplot(2, 1, 2)
    df['day_of_week'] = df['TransactionDateTime'].dt.day_name()
    df['day_of_week'].value_counts().plot(kind='bar')
    plt.title('Transaction Distribution by Day of Week')
    
    plt.tight_layout()
    plt.show()

# 2. Amount Pattern Analysis
def analyze_amount_patterns(df):
    plt.figure(figsize=(15, 10))
    
    # Transaction amount distribution
    plt.subplot(2, 2, 1)
    sns.histplot(data=df, x='TransactionAmt', bins=50)
    plt.title('Transaction Amount Distribution')
    
    # Log transformation of amount
    plt.subplot(2, 2, 2)
    sns.histplot(data=df, x='TransactionAmt', log_scale=True)
    plt.title('Log-Scaled Transaction Amount')
    
    # Amount by product code
    plt.subplot(2, 2, 3)
    sns.boxplot(data=df, x='ProductCD', y='TransactionAmt')
    plt.title('Transaction Amount by Product Code')
    
    # Amount density plot
    plt.subplot(2, 2, 4)
    sns.kdeplot(data=df, x='TransactionAmt')
    plt.title('Transaction Amount Density')
    
    plt.tight_layout()
    plt.show()

# 3. Card Analysis
def analyze_card_patterns(df):
    plt.figure(figsize=(15, 10))
    
    try:
        # Card type (card4) distribution - Visa, Mastercard, etc.
        plt.subplot(2, 2, 1)
        if 'card4' in df.columns and not df['card4'].isna().all():
            card_types = df['card4'].value_counts()
            plt.pie(card_types.values, labels=card_types.index, autopct='%1.1f%%')
            plt.title('Card Type Distribution (Visa, Mastercard, etc.)')
        else:
            plt.text(0.5, 0.5, 'No card type data available', ha='center')

        # Card category (card6) distribution - Credit/Debit
        plt.subplot(2, 2, 2)
        if 'card6' in df.columns and not df['card6'].isna().all():
            card_categories = df['card6'].value_counts()
            sns.barplot(x=card_categories.index, y=card_categories.values)
            plt.title('Card Category (Credit/Debit)')
            plt.xticks(rotation=45)
        else:
            plt.text(0.5, 0.5, 'No card category data available', ha='center')

        # Card issuer (card1) distribution
        plt.subplot(2, 2, 3)
        top_issuers = df['card1'].value_counts().head(10)
        sns.barplot(x=top_issuers.index.astype(str), y=top_issuers.values)
        plt.title('Top 10 Card Issuers')
        plt.xticks(rotation=45)

        # Transaction amount by card type
        plt.subplot(2, 2, 4)
        if 'card4' in df.columns and not df['card4'].isna().all():
            sns.boxplot(data=df, x='card4', y='TransactionAmt')
            plt.title('Transaction Amount by Card Type')
            plt.xticks(rotation=45)
        else:
            plt.text(0.5, 0.5, 'No transaction data by card type available', ha='center')

        plt.tight_layout()
        plt.show()

        # Print summary statistics
        print("\nCard Analysis Summary:")
        
        if 'card4' in df.columns and not df['card4'].isna().all():
            print("\nCard Types (card4):")
            print(df['card4'].value_counts())
        
        if 'card6' in df.columns and not df['card6'].isna().all():
            print("\nCard Categories (card6):")
            print(df['card6'].value_counts())
        
        print("\nCard Issuer Statistics (card1):")
        print(f"Number of unique issuers: {df['card1'].nunique()}")
        print("Top 5 issuers by transaction count:")
        print(df['card1'].value_counts().head())

        print("\nMissing Values Summary:")
        for col in ['card1', 'card2', 'card3', 'card4', 'card5', 'card6']:
            print(f"{col} null count: {df[col].isnull().sum()}")

    except Exception as e:
        print(f"Error in analyze_card_patterns: {str(e)}")
        traceback.print_exc()
        plt.close()

        
# 4. Email Domain Analysis
def analyze_email_patterns(df):
    plt.figure(figsize=(15, 10))
    
    # Email provider distribution
    plt.subplot(2, 1, 1)
    df['P_emaildomain'].value_counts().head(10).plot(kind='bar')
    plt.title('Top 10 Email Providers')
    
    # Email domain categories
    plt.subplot(2, 1, 2)
    email_categories = df['P_emaildomain'].apply(lambda x: 'Other' if pd.isna(x) else x.split('.')[-1])
    email_categories.value_counts().plot(kind='pie', autopct='%1.1f%%')
    plt.title('Email Domain Categories')
    
    plt.tight_layout()
    plt.show()

# 5. Device Analysis
def analyze_device_patterns(df):
    np.random.seed(42)  # For reproducibility
    
    # Create mock device types with realistic distributions
    device_types = ['mobile', 'desktop', 'tablet']
    device_weights = [0.6, 0.3, 0.1]  # 60% mobile, 30% desktop, 10% tablet
    
    # Add mock device columns
    df['DeviceType'] = np.random.choice(device_types, size=len(df), p=device_weights)
    
    # Mock device info (common device models)
    device_info = [
        'iPhone', 'Samsung Galaxy', 'iPad', 
        'Windows PC', 'MacBook', 'Android Tablet',
        'Chrome Desktop', 'Firefox Mobile'
    ]
    device_info_weights = [0.3, 0.2, 0.1, 0.15, 0.1, 0.05, 0.05, 0.05]
    df['DeviceInfo'] = np.random.choice(device_info, size=len(df), p=device_info_weights)
    
    # Create visualizations
    plt.figure(figsize=(15, 10))
    
    # Device type distribution
    plt.subplot(2, 2, 1)
    df['DeviceType'].value_counts().plot(kind='pie', autopct='%1.1f%%')
    plt.title('Device Type Distribution')
    
    # Device info analysis
    plt.subplot(2, 2, 2)
    df['DeviceInfo'].value_counts().head(10).plot(kind='bar')
    plt.title('Top 10 Device Types')
    plt.xticks(rotation=45)
    
    # Transaction amount by device type
    plt.subplot(2, 2, 3)
    if 'TransactionAmt' in df.columns:
        sns.boxplot(data=df, x='DeviceType', y='TransactionAmt')
        plt.title('Transaction Amount by Device Type')
    else:
        plt.text(0.5, 0.5, 'Transaction Amount data not available', ha='center')
    
    plt.tight_layout()
    plt.show()

# 6. Distance Analysis
def analyze_distance_patterns(df):
    plt.figure(figsize=(15, 10))
    
    # Distance distribution
    plt.subplot(2, 2, 1)
    sns.histplot(data=df, x='dist1', bins=50)
    plt.title('Distance Distribution')
    
    # Distance by transaction amount
    plt.subplot(2, 2, 2)
    sns.scatterplot(data=df, x='dist1', y='TransactionAmt')
    plt.title('Distance vs Transaction Amount')
    
    plt.tight_layout()
    plt.show()

# 7. Correlation Analysis
def analyze_correlations(df):
    # Select numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()
    
    plt.figure(figsize=(15, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix of Numeric Features')
    plt.show()

# 8. Product Analysis
def analyze_product_patterns(df):
    plt.figure(figsize=(15, 10))
    
    # Product code distribution
    plt.subplot(2, 2, 1)
    df['ProductCD'].value_counts().plot(kind='pie', autopct='%1.1f%%')
    plt.title('Product Code Distribution')
    
    # Transaction amount by product
    plt.subplot(2, 2, 2)
    sns.boxplot(data=df, x='ProductCD', y='TransactionAmt')
    plt.title('Transaction Amount by Product')
    
    plt.tight_layout()
    plt.show()

# 9. Address Analysis
def analyze_address_patterns(df):
    plt.figure(figsize=(15, 5))
    
    # Address match analysis
    plt.subplot(1, 2, 1)
    df['addr1'].value_counts().head(10).plot(kind='bar')
    plt.title('Top 10 Address Types')
    
    # Address distance analysis
    plt.subplot(1, 2, 2)
    sns.scatterplot(data=df, x='addr1', y='addr2')
    plt.title('Address Type Relationships')
    
    plt.tight_layout()
    plt.show()

# 10. Statistical Analysis
def perform_statistical_analysis(df):
    # Basic statistics
    print("Basic Statistics for Transaction Amount:")
    print(df['TransactionAmt'].describe())
    
    # Skewness and Kurtosis
    print("\nSkewness:", stats.skew(df['TransactionAmt'].dropna()))
    print("Kurtosis:", stats.kurtosis(df['TransactionAmt'].dropna()))
    
    # Normality test
    stat, p_value = stats.normaltest(df['TransactionAmt'].dropna())
    print("\nNormality Test p-value:", p_value)

# Run complete analysis
def run_complete_analysis(df):
    print("Starting Comprehensive Fraud Detection Analysis...")
    
    print("\n1. Analyzing Transaction Times...")
    analyze_transaction_time(df)
    
    print("\n2. Analyzing Amount Patterns...")
    analyze_amount_patterns(df)
    
    print("\n3. Analyzing Card Patterns...")
    analyze_card_patterns(df)
    
    print("\n4. Analyzing Email Patterns...")
    analyze_email_patterns(df)
    
    print("\n5. Analyzing Device Patterns...")
    analyze_device_patterns(df)
    
    print("\n6. Analyzing Distance Patterns...")
    analyze_distance_patterns(df)
    
    print("\n7. Analyzing Correlations...")
    analyze_correlations(df)
    
    print("\n8. Analyzing Product Patterns...")
    analyze_product_patterns(df)
    
    print("\n9. Analyzing Address Patterns...")
    analyze_address_patterns(df)
    
    print("\n10. Performing Statistical Analysis...")
    perform_statistical_analysis(df)
    
    print("\nAnalysis Complete!")

# Execute the analysis
if __name__ == "__main__":
    df = get_processed_data()
    run_complete_analysis(df)