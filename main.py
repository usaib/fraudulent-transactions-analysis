import traceback
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
from data_processing.data_processing import ParallelDynamoProcessor, parallel_process_data

# Load environment variables
load_dotenv()

app = Flask(__name__)
def format_fraud_analysis(df):
    """Create formatted analysis data for fraud detection visualizations"""
    results = []
    
    try:
         # Ensure numeric columns are properly typed
        numeric_columns = ['TransactionAmt', 'dist1', 'card1', 'addr1', 'addr2']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        # 1. Transactions by Hour
        hourly_trans = df.groupby(df['TransactionDateTime'].dt.hour).size()
        results.append({
            'graphName': 'hourly_distribution',
            'graphType': 'line',
            'values': {
                'title': 'Transaction Distribution by Hour',
                'labels': [f"{h:02d}:00" for h in hourly_trans.index],
                'datasets': [{
                    'label': 'Number of Transactions',
                    'data': hourly_trans.tolist()
                }]
            }
        })

        # 2. Transactions by Day of Week
        daily_trans = df.groupby(df['TransactionDateTime'].dt.day_name()).size()
        results.append({
            'graphName': 'daily_distribution',
            'graphType': 'bar',
            'values': {
                'title': 'Transaction Distribution by Day',
                'labels': daily_trans.index.tolist(),
                'datasets': [{
                    'label': 'Number of Transactions',
                    'data': daily_trans.values.tolist()
                }]
            }
        })

        # 3. Transaction Amount Distribution
        amount_bins = np.histogram(df['TransactionAmt'], bins=50)
        results.append({
            'graphName': 'amount_distribution',
            'graphType': 'histogram',
            'values': {
                'title': 'Transaction Amount Distribution',
                'labels': amount_bins[1][:-1].tolist(),
                'datasets': [{
                    'label': 'Frequency',
                    'data': amount_bins[0].tolist()
                }]
            }
        })

        # 4. Log-Scaled Transaction Amount
        log_amount_bins = np.histogram(np.log1p(df['TransactionAmt']), bins=50)
        results.append({
            'graphName': 'log_amount_distribution',
            'graphType': 'histogram',
            'values': {
                'title': 'Log-Scaled Transaction Amount',
                'labels': log_amount_bins[1][:-1].tolist(),
                'datasets': [{
                    'label': 'Frequency',
                    'data': log_amount_bins[0].tolist()
                }]
            }
        })

        # 5. Amount by Product Code
        amt_by_product = df.groupby('ProductCD')['TransactionAmt'].agg(['mean', 'min', 'max', 'median']).round(2)
        results.append({
            'graphName': 'amount_by_product',
            'graphType': 'boxplot',
            'values': {
                'title': 'Transaction Amount by Product Code',
                'labels': amt_by_product.index.tolist(),
                'datasets': [{
                    'label': 'Amount Statistics',
                    'data': amt_by_product.to_dict('records')
                }]
            }
        })

        # 6. Card Type Distribution (card4)
        if 'card4' in df.columns and not df['card4'].isna().all():
            card_types = df['card4'].value_counts()
            results.append({
                'graphName': 'card_type_distribution',
                'graphType': 'pie',
                'values': {
                    'title': 'Card Type Distribution',
                    'labels': card_types.index.tolist(),
                    'datasets': [{
                        'label': 'Percentage',
                        'data': (card_types / card_types.sum() * 100).round(1).tolist()
                    }]
                }
            })

        # 7. Card Category Distribution (card6)
        if 'card6' in df.columns and not df['card6'].isna().all():
            card_categories = df['card6'].value_counts()
            results.append({
                'graphName': 'card_category_distribution',
                'graphType': 'bar',
                'values': {
                    'title': 'Card Category Distribution',
                    'labels': card_categories.index.tolist(),
                    'datasets': [{
                        'label': 'Count',
                        'data': card_categories.values.tolist()
                    }]
                }
            })

        # 8. Card Issuer Distribution (card1)
        top_issuers = df['card1'].value_counts().head(10)
        results.append({
            'graphName': 'card_issuer_distribution',
            'graphType': 'bar',
            'values': {
                'title': 'Top 10 Card Issuers',
                'labels': top_issuers.index.astype(str).tolist(),
                'datasets': [{
                    'label': 'Count',
                    'data': top_issuers.values.tolist()
                }]
            }
        })

        # 9. Email Provider Distribution
        email_providers = df['P_emaildomain'].value_counts().head(10)
        results.append({
            'graphName': 'email_provider_distribution',
            'graphType': 'bar',
            'values': {
                'title': 'Top 10 Email Providers',
                'labels': email_providers.index.tolist(),
                'datasets': [{
                    'label': 'Count',
                    'data': email_providers.values.tolist()
                }]
            }
        })

        # 10. Email Domain Categories
        email_categories = df['P_emaildomain'].apply(
            lambda x: 'Other' if pd.isna(x) else x.split('.')[-1]
        ).value_counts()
        results.append({
            'graphName': 'email_domain_categories',
            'graphType': 'pie',
            'values': {
                'title': 'Email Domain Categories',
                'labels': email_categories.index.tolist(),
                'datasets': [{
                    'label': 'Percentage',
                    'data': (email_categories / email_categories.sum() * 100).round(1).tolist()
                }]
            }
        })

        # 11. Distance Analysis
        if 'dist1' in df.columns:
            distance_bins = np.histogram(df['dist1'].dropna(), bins=50)
            results.append({
                'graphName': 'distance_distribution',
                'graphType': 'histogram',
                'values': {
                    'title': 'Distance Distribution',
                    'labels': distance_bins[1][:-1].tolist(),
                    'datasets': [{
                        'label': 'Frequency',
                        'data': distance_bins[0].tolist()
                    }]
                }
            })

        # 12. Correlation Analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr_matrix = df[numeric_cols].corr().round(2)
        results.append({
            'graphName': 'correlation_matrix',
            'graphType': 'heatmap',
            'values': {
                'title': 'Correlation Matrix',
                'labels': numeric_cols.tolist(),
                'datasets': [{
                    'label': 'Correlation',
                    'data': corr_matrix.values.tolist()
                }]
            }
        })

        # 13. Address Analysis
        top_addresses = df['addr1'].value_counts().head(10)
        results.append({
            'graphName': 'address_distribution',
            'graphType': 'bar',
            'values': {
                'title': 'Top 10 Address Types',
                'labels': top_addresses.index.astype(str).tolist(),
                'datasets': [{
                    'label': 'Count',
                    'data': top_addresses.values.tolist()
                }]
            }
        })

        # 14. Statistical Summary
        stats_summary = df['TransactionAmt'].describe()
        results.append({
            'graphName': 'statistical_summary',
            'graphType': 'stats',
            'values': {
                'title': 'Transaction Amount Statistics',
                'stats': {
                    'count': int(stats_summary['count']),
                    'mean': round(stats_summary['mean'], 2),
                    'std': round(stats_summary['std'], 2),
                    'min': round(stats_summary['min'], 2),
                    '25%': round(stats_summary['25%'], 2),
                    'median': round(stats_summary['50%'], 2),
                    '75%': round(stats_summary['75%'], 2),
                    'max': round(stats_summary['max'], 2),
                    'skewness': round(df['TransactionAmt'].skew(), 2),
                    'kurtosis': round(df['TransactionAmt'].kurtosis(), 2)
                }
            }
        })

         # Product Code Analyses
        product_dist = df['ProductCD'].value_counts()
        results.append({
            'graphName': 'product_code_distribution',
            'graphType': 'pie',
            'values': {
                'title': 'Product Code Distribution',
                'labels': product_dist.index.tolist(),
                'datasets': [{
                    'label': 'Percentage',
                    'data': (product_dist / product_dist.sum() * 100).round(1).tolist()
                }]
            }
        })

        # Transaction Amount by Product
        product_amount_stats = df.groupby('ProductCD')['TransactionAmt'].agg([
            'mean', 'median', 'std', 'min', 'max'
        ]).round(2)
        results.append({
            'graphName': 'amount_by_product_boxplot',
            'graphType': 'boxplot',
            'values': {
                'title': 'Transaction Amount by Product',
                'labels': product_amount_stats.index.tolist(),
                'datasets': [{
                    'label': 'Amount Statistics',
                    'data': product_amount_stats.to_dict('records')
                }]
            }
        })

        # Device Analysis (with mock data)
        np.random.seed(42)
        
        # Add mock device types
        device_types = ['mobile', 'desktop', 'tablet']
        device_weights = [0.6, 0.3, 0.1]
        df['DeviceType'] = np.random.choice(device_types, size=len(df), p=device_weights)
        
        # Add mock device info
        device_info = [
            'iPhone', 'Samsung Galaxy', 'iPad', 
            'Windows PC', 'MacBook', 'Android Tablet',
            'Chrome Desktop', 'Firefox Mobile'
        ]
        device_info_weights = [0.3, 0.2, 0.1, 0.15, 0.1, 0.05, 0.05, 0.05]
        df['DeviceInfo'] = np.random.choice(device_info, size=len(df), p=device_info_weights)

        # Device Type Distribution
        device_type_dist = df['DeviceType'].value_counts()
        results.append({
            'graphName': 'device_type_distribution',
            'graphType': 'pie',
            'values': {
                'title': 'Device Type Distribution',
                'labels': device_type_dist.index.tolist(),
                'datasets': [{
                    'label': 'Percentage',
                    'data': (device_type_dist / device_type_dist.sum() * 100).round(1).tolist()
                }]
            }
        })

        # Device Info Analysis
        device_info_dist = df['DeviceInfo'].value_counts().head(10)
        results.append({
            'graphName': 'device_info_distribution',
            'graphType': 'bar',
            'values': {
                'title': 'Top 10 Device Types',
                'labels': device_info_dist.index.tolist(),
                'datasets': [{
                    'label': 'Count',
                    'data': device_info_dist.values.tolist()
                }]
            }
        })

        # Transaction Amount by Device Type
        device_amount_stats = df.groupby('DeviceType')['TransactionAmt'].agg([
            'mean', 'median', 'std', 'min', 'max'
        ]).round(2)
        results.append({
            'graphName': 'amount_by_device_boxplot',
            'graphType': 'boxplot',
            'values': {
                'title': 'Transaction Amount by Device Type',
                'labels': device_amount_stats.index.tolist(),
                'datasets': [{
                    'label': 'Amount Statistics',
                    'data': device_amount_stats.to_dict('records')
                }]
            }
        })

        return results

    except Exception as e:
        print(f"Error in analysis: {str(e)}")
        return []
@app.route('/api/fraud-data', methods=['POST'])
def get_fraud_data():
    try:
        payload = request.json
        segment_count = int(payload.get('segment_count', 8))
        start_date = payload.get('start_date')
        end_date = payload.get('end_date')
        limit = int(payload.get('limit', 1000))

        # Initialize parallel processor
        processor = ParallelDynamoProcessor('Fraud_Detection', segment_count=segment_count)
        
        # Fetch data
        df = processor.fetch_data_parallel()
        
        # Process data
        df_processed = parallel_process_data(df)
        
        # Convert TransactionDT to actual datetime
        reference_date = pd.Timestamp('2017-01-01')
        if 'TransactionDT' in df_processed.columns:
            # First ensure TransactionDT is numeric
            df_processed['TransactionDT'] = pd.to_numeric(df_processed['TransactionDT'], errors='coerce')
            # Then convert to timedelta and add to reference date
            df_processed['TransactionDateTime'] = reference_date + pd.to_timedelta(df_processed['TransactionDT'].astype(float), unit='s')
        
        # Apply date filters if provided
        if start_date or end_date:
            if start_date:
                start_datetime = pd.to_datetime(start_date).tz_localize(None)
                df_processed = df_processed[df_processed['TransactionDateTime'] >= start_datetime]
            if end_date:
                end_datetime = pd.to_datetime(end_date).tz_localize(None)
                df_processed = df_processed[df_processed['TransactionDateTime'] <= end_datetime]
        
        # Limit results if specified
        if limit:
            df_processed = df_processed.head(limit)

        # Generate analysis
        analysis_results = format_fraud_analysis(df_processed)

        # Prepare raw data for JSON serialization
        # Convert datetime objects to ISO format strings
        df_processed['TransactionDateTime'] = df_processed['TransactionDateTime'].dt.strftime('%Y-%m-%dT%H:%M:%S')
        raw_data = df_processed.to_dict(orient='records')
        
        # Get column information
        columns = [
            {
                'name': col,
                'type': str(df_processed[col].dtype),
                'unique_values': df_processed[col].nunique(),
                'is_datetime': 'datetime' in str(df_processed[col].dtype).lower()
            }
            for col in df_processed.columns
        ]

        return jsonify({
            'status': 'success',
            'data': {
                'raw_data': raw_data,
                'columns': columns,
                'analysis': analysis_results,
                'metadata': {
                    'total_records': len(df_processed),
                    'segment_count': segment_count,
                    'processing_time': processor.processing_time if hasattr(processor, 'processing_time') else None,
                    'reference_date': reference_date.strftime('%Y-%m-%d'),
                    'date_range': {
                        'min': df_processed['TransactionDateTime'].min(),
                        'max': df_processed['TransactionDateTime'].max()
                    }
                }
            }
        })

    except Exception as e:
        print(f"Error in get_fraud_data: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/',methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

if __name__ == '__main__':
    app.run(debug=True)