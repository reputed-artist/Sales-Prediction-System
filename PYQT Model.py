import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# PyQt5 imports
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                            QLabel, QTableWidget, QTableWidgetItem, QHeaderView,
                            QMessageBox, QProgressBar, QTextEdit)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QColor, QBrush
import sys

class SalesPredictionEngine:
    def __init__(self):
        self.model = None
        self.feature_columns = None
        self.model_performance = {}
        self.client_history = {}
    def identify_frequent_buyers(self, df):
        """Identify frequent buyers based on multiple criteria"""
        df_processed = self.prepare_data(df)
        
        if df_processed.empty:
            return pd.DataFrame()
            
        client_stats = df_processed.groupby('client name').agg({
            'days_diff': ['mean', 'count'],
            'invdate': ['min', 'max']
        }).round(2)
        
        # Flatten column names
        client_stats.columns = ['avg_gap', 'total_purchases', 'first_purchase', 'last_purchase']
        
        # Calculate additional metrics
        client_stats['days_since_last'] = (datetime.now() - client_stats['last_purchase']).dt.days
        
        # Calculate purchase frequency (purchases per month)
        total_days = (client_stats['last_purchase'] - client_stats['first_purchase']).dt.days + 1
        client_stats['purchase_frequency'] = (client_stats['total_purchases'] / (total_days / 30)).round(2)  # per month
        
        # Categorize clients by frequency
        conditions = [
            (client_stats['avg_gap'] <= 7) & (client_stats['total_purchases'] >= 4),
            (client_stats['avg_gap'] <= 15) & (client_stats['total_purchases'] >= 3),
            (client_stats['avg_gap'] <= 30) & (client_stats['total_purchases'] >= 2),
        ]
        
        categories = ['FREQUENT_WEEKLY', 'FREQUENT_BIWEEKLY', 'REGULAR_MONTHLY']
        client_stats['buyer_type'] = np.select(conditions, categories, default='OCCASIONAL')
        
        # Add priority level
        priority_map = {
            'FREQUENT_WEEKLY': 'HIGH',
            'FREQUENT_BIWEEKLY': 'HIGH',
            'REGULAR_MONTHLY': 'MEDIUM',
            'OCCASIONAL': 'LOW'
        }
        client_stats['priority'] = client_stats['buyer_type'].map(priority_map)
        
        return client_stats.sort_values('purchase_frequency', ascending=False)    
    def clean_data(self, df):
        """Clean and validate the input data"""
        # Make a copy to avoid modifying original
        df_clean = df.copy()
        
        # Remove summary rows (like "Total") and invalid dates
        date_mask = pd.to_datetime(df_clean["invdate"], errors='coerce', dayfirst=True).notna()
        df_clean = df_clean[date_mask]
        
        # Remove rows with missing critical data
        df_clean = df_clean.dropna(subset=["client name", "invdate"])
        
        # Clean column names (remove extra spaces)
        df_clean.columns = df_clean.columns.str.strip()
        
        print(f"Cleaned data: {len(df_clean)} rows remaining from original {len(df)}")
        return df_clean
    
    def prepare_data(self, df):
        """Enhanced data preparation with more features"""
        # Clean data first
        df_clean = self.clean_data(df)
        
        if df_clean.empty:
            return df_clean
            
        # Convert date with error handling
        df_clean["invdate"] = pd.to_datetime(df_clean["invdate"], errors='coerce', dayfirst=True)
        
        # Remove any remaining invalid dates
        df_clean = df_clean[df_clean["invdate"].notna()]
        
        # Sort by client and date
        df_clean = df_clean.sort_values(by=["client name", "invdate"])
        
        # Calculate days between purchases
        df_clean["days_diff"] = df_clean.groupby("client name")["invdate"].diff().dt.days
        
        # Remove first row for each client (no previous purchase) and invalid gaps
        df_clean = df_clean.dropna(subset=["days_diff"])
        df_clean = df_clean[df_clean["days_diff"] > 0]  # Remove negative or zero gaps
        
        # Create additional features
        df_clean = self.create_additional_features(df_clean)
        
        return df_clean
    
    def create_additional_features(self, df):
        """Create more meaningful features for prediction"""
        # Client purchase patterns
        client_stats = df.groupby("client name").agg({
            'days_diff': ['mean', 'std', 'min', 'max'],
            'invdate': 'count'
        }).round(2)
        
        # Flatten column names
        client_stats.columns = ['_'.join(col).strip() for col in client_stats.columns.values]
        client_stats = client_stats.rename(columns={
            'days_diff_mean': 'avg_purchase_gap',
            'days_diff_std': 'std_purchase_gap',
            'days_diff_min': 'min_purchase_gap',
            'days_diff_max': 'max_purchase_gap',
            'invdate_count': 'total_purchases'
        })
        
        df = df.merge(client_stats, on="client name", how='left')
        
        # Time-based features
        df['day_of_week'] = df['invdate'].dt.dayofweek
        df['month'] = df['invdate'].dt.month
        df['quarter'] = df['invdate'].dt.quarter
        df['day_of_month'] = df['invdate'].dt.day
        df['year'] = df['invdate'].dt.year
        
        # Seasonal features
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_month_start'] = (df['day_of_month'] == 1).astype(int)
        df['is_month_end'] = (df['day_of_month'] >= 28).astype(int)
        
        # Cyclical features
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Rolling statistics
        df['prev_gap'] = df.groupby("client name")['days_diff'].shift(1)
        df['gap_trend'] = df['days_diff'] - df['prev_gap']
        
        # Fill NaN values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(method='bfill').fillna(method='ffill')
        
        return df
    
    def train_model(self, df):
        """Train the prediction model with enhanced features"""
        try:
            # Prepare data
            df_processed = self.prepare_data(df)
            
            if len(df_processed) < 5:
                return False, f"Not enough data for training (minimum 5 valid records required, found {len(df_processed)})"
            
            # One-hot encoding for categorical variables
            categorical_cols = ["location", "item name", "client name"]
            # Only use categorical columns that exist in the dataframe
            categorical_cols = [col for col in categorical_cols if col in df_processed.columns]
            
            df_encoded = pd.get_dummies(df_processed, columns=categorical_cols, drop_first=True)
            
            # Define features and target
            exclude_cols = ["invid", "invdate", "days_diff", "sr no"]
            # Only exclude columns that exist
            exclude_cols = [col for col in exclude_cols if col in df_encoded.columns]
            
            feature_cols = [col for col in df_encoded.columns if col not in exclude_cols and col != "days_diff"]
            
            # Check if we have enough features
            if len(feature_cols) == 0:
                return False, "No features available for training"
            
            X = df_encoded[feature_cols]
            y = df_encoded["days_diff"]
            
            self.feature_columns = X.columns.tolist()
            
            # Split data - use smaller test size if limited data
            test_size = min(0.2, 0.3) if len(X) < 50 else 0.2
            if len(X) < 10:
                X_train, X_test, y_train, y_test = X, X[:0], y, y[:0]  # No test split for very small data
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )
            
            # Train model with parameters suitable for small datasets
            self.model = RandomForestRegressor(
                n_estimators=50 if len(X_train) < 50 else 100,
                max_depth=5 if len(X_train) < 30 else 8,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42
            )
            
            self.model.fit(X_train, y_train)
            
            # Evaluate model
            train_score = self.model.score(X_train, y_train)
            test_score = self.model.score(X_test, y_test) if len(X_test) > 0 else 0
            
            y_pred = self.model.predict(X_test) if len(X_test) > 0 else []
            mae = mean_absolute_error(y_test, y_pred) if len(y_test) > 0 else 0
            rmse = np.sqrt(mean_squared_error(y_test, y_pred)) if len(y_test) > 0 else 0
            
            self.model_performance = {
                'train_r2': train_score,
                'test_r2': test_score,
                'mae': mae,
                'rmse': rmse,
                'training_samples': len(X_train),
                'total_clients': df_processed["client name"].nunique(),
                'date_range': f"{df_processed['invdate'].min().strftime('%Y-%m-%d')} to {df_processed['invdate'].max().strftime('%Y-%m-%d')}",
                'last_trained': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Store client history for future predictions
            self.update_client_history(df_processed)
            
            return True, f"Model trained successfully with {len(X_train)} samples from {df_processed['client name'].nunique()} clients"
            
        except Exception as e:
            return False, f"Training failed: {str(e)}"
    
    def update_client_history(self, df):
        """Store client purchase history for feature generation"""
        for client, group in df.groupby("client name"):
            self.client_history[client] = {
                'last_purchase': group['invdate'].max(),
                'purchase_count': len(group),
                'avg_gap': group['days_diff'].mean(),
                'recent_gaps': group['days_diff'].tail(3).tolist()
            }
    
        
    def predict_next_sale(self, df):
        """Predict next sale date for all clients considering TODAY'S date"""
        if self.model is None:
            return None, "Model not trained"
        
        try:
            # ===== DEBUGGING: Track client counts =====
            total_unique_clients = df["client name"].nunique()
            print(f"🔍 DEBUG: Total unique clients in raw data: {total_unique_clients}")
            
            # Process the input data
            df_processed = self.prepare_data(df)
            
            clients_after_prepare = df_processed["client name"].nunique() if not df_processed.empty else 0
            print(f"🔍 DEBUG: Clients after prepare_data(): {clients_after_prepare}")
            
            if len(df_processed) == 0:
                return None, "No valid data for prediction"
            
            categorical_cols = ["location", "item name", "client name"]
            categorical_cols = [col for col in categorical_cols if col in df_processed.columns]
            
            df_encoded = pd.get_dummies(df_processed, columns=categorical_cols, drop_first=True)
            
            predictions = []
            today = datetime.now().date()
            
            # ===== DEBUGGING: Track exclusion reasons =====
            exclusion_stats = {
                'single_purchase': 0,
                'encoding_failed': 0,
                'total_processed': 0
            }
            
            for client, group in df_processed.groupby("client name"):
                exclusion_stats['total_processed'] += 1
                
                if len(group) < 2:  # Need at least 2 purchases for prediction
                    exclusion_stats['single_purchase'] += 1
                    continue
                    
                last_row = group.iloc[-1]
                last_date = last_row["invdate"].date()
                
                # Calculate how many days have passed since last invoice
                days_since_last = (today - last_date).days
                
                # Find corresponding row in encoded dataframe
                client_encoded = df_encoded[df_encoded.index == last_row.name]
                
                if len(client_encoded) == 0:
                    exclusion_stats['encoding_failed'] += 1
                    continue
                
                # Ensure all feature columns are present
                for col in self.feature_columns:
                    if col not in client_encoded.columns:
                        client_encoded[col] = 0
                
                # Ensure correct feature order
                client_features = client_encoded[self.feature_columns]
                
                # Predict days until next purchase
                predicted_gap = self.model.predict(client_features)[0]
                predicted_gap = max(1, predicted_gap)
                
                # Intelligent prediction considering today's date
                if days_since_last > predicted_gap:
                    # Client is overdue - predict sooner
                    days_overdue = days_since_last - predicted_gap
                    if days_overdue > 30:
                        adjusted_gap = 3  # Very overdue - predict in 3 days
                    elif days_overdue > 14:
                        adjusted_gap = 7  # Moderately overdue - predict in 7 days
                    else:
                        adjusted_gap = max(1, predicted_gap * 0.5)
                    
                    predicted_date = today + timedelta(days=adjusted_gap)
                    status = f"OVERDUE ({days_overdue} days)"
                else:
                    # Normal case
                    days_until_next = predicted_gap - days_since_last
                    predicted_date = today + timedelta(days=int(round(days_until_next)))
                    status = "ON_TRACK"
                
                # Determine buyer type
                if predicted_gap <= 7 and len(group) >= 4:
                    buyer_type = "FREQUENT_WEEKLY"
                    priority = "HIGH"
                elif predicted_gap <= 15 and len(group) >= 3:
                    buyer_type = "FREQUENT_BIWEEKLY"
                    priority = "HIGH"
                elif predicted_gap <= 30 and len(group) >= 2:
                    buyer_type = "REGULAR_MONTHLY"
                    priority = "MEDIUM"
                else:
                    buyer_type = "OCCASIONAL"
                    priority = "LOW"
                
                # Calculate confidence based on client history
                confidence = self.calculate_confidence(client, predicted_gap)
                
                predictions.append({
                    'client': client,
                    'last_invoice': last_date.strftime("%Y-%m-%d"),
                    'days_since_last': days_since_last,
                    'typical_gap': round(predicted_gap, 1),
                    'predicted_next_sale': predicted_date.strftime("%Y-%m-%d"),
                    'confidence': f"{confidence:.1%}",
                    'days_until_next': (predicted_date - today).days,
                    'purchase_count': len(group),
                    'status': status,
                    'buyer_type': buyer_type,
                    'priority': priority
                })
            
            # ===== DEBUGGING: Print exclusion analysis =====
            print(f"\n🔍 DEBUG: EXCLUSION ANALYSIS")
            print(f"Total clients processed: {exclusion_stats['total_processed']}")
            print(f"✅ Clients with predictions: {len(predictions)}")
            print(f"❌ Excluded clients: {exclusion_stats['single_purchase'] + exclusion_stats['encoding_failed']}")
            print(f"   - Single purchase clients: {exclusion_stats['single_purchase']}")
            print(f"   - Encoding failed: {exclusion_stats['encoding_failed']}")
            
            if not predictions:
                return None, "No predictions could be generated"
                
            # Convert to DataFrame and add priority score
            result_df = pd.DataFrame(predictions)
            result_df = self.add_priority_scores(result_df)
            
            # Sort by priority score (highest first)
            result_df = result_df.sort_values('priority_score', ascending=False)
            
            return result_df, "Success"
            
        except Exception as e:
            return None, f"Prediction failed: {str(e)}"
    
    def add_priority_scores(self, predictions_df):
        """Calculate priority scores for clients with safe type handling"""
        def calculate_priority_score(row):
            score = 0
            
            # Buyer type scoring
            if row['buyer_type'] == 'FREQUENT_WEEKLY':
                score += 30
            elif row['buyer_type'] == 'FREQUENT_BIWEEKLY':
                score += 25
            elif row['buyer_type'] == 'REGULAR_MONTHLY':
                score += 15
            else:
                score += 5
            
            # Days until next sale scoring
            try:
                days_until_next = int(float(row['days_until_next']))  # Handle float values
                if days_until_next <= 7:
                    score += 25
                elif days_until_next <= 14:
                    score += 15
                elif days_until_next <= 30:
                    score += 10
                else:
                    score += 5
            except (ValueError, TypeError):
                pass  # Skip if conversion fails
            
            # Confidence scoring
            try:
                confidence_str = str(row['confidence']).strip('%')
                confidence = float(confidence_str) / 100
                score += int(confidence * 20)
            except (ValueError, AttributeError):
                pass  # Skip if conversion fails
            
            # Purchase count scoring
            try:
                purchase_count = int(float(row['purchase_count']))  # Handle float values
                if purchase_count >= 5:
                    score += 10
                elif purchase_count >= 3:
                    score += 5
            except (ValueError, TypeError):
                pass  # Skip if conversion fails
            
            return score
        
        predictions_df['priority_score'] = predictions_df.apply(calculate_priority_score, axis=1)
        return predictions_df   
 
    def identify_potential_buyers(self, predictions_df):
        """Identify potential buyers using multiple criteria with safe type handling"""
        
        potential_buyers = []
        
        for _, row in predictions_df.iterrows():
            score = 0
            reasons = []
            
            # 1. PRIORITY LEVEL (Most Important)
            if row['priority'] == 'HIGH':
                score += 30
                reasons.append("High priority client")
            elif row['priority'] == 'MEDIUM':
                score += 15
                reasons.append("Medium priority client")
            
            # 2. STATUS (Overdue clients are hot leads)
            if 'OVERDUE' in row['status']:
                score += 25
                try:
                    # Extract overdue days safely
                    status_str = str(row['status'])
                    if '(' in status_str and 'days' in status_str:
                        overdue_part = status_str.split('(')[1].split(' ')[0]
                        overdue_days = int(float(overdue_part))  # Handle float strings
                        reasons.append(f"Overdue by {overdue_days} days")
                    else:
                        reasons.append("Overdue client")
                except (ValueError, IndexError):
                    reasons.append("Overdue client")
            elif row['status'] == 'ON_TRACK':
                score += 10
                reasons.append("Regular buying pattern")
            
            # 3. CONFIDENCE LEVEL
            try:
                confidence_str = str(row['confidence']).strip('%')
                confidence = float(confidence_str) / 100
                if confidence >= 0.8:
                    score += 20
                    reasons.append("High prediction confidence")
                elif confidence >= 0.6:
                    score += 10
                    reasons.append("Good prediction confidence")
            except (ValueError, AttributeError):
                pass  # Skip confidence scoring if conversion fails
            
            # 4. DAYS UNTIL NEXT SALE (Soon = More Potential)
            try:
                days_until_next = int(float(row['days_until_next']))  # Handle float values
                if days_until_next <= 7:
                    score += 25
                    reasons.append("Expected within 7 days")
                elif days_until_next <= 14:
                    score += 15
                    reasons.append("Expected within 14 days")
                elif days_until_next <= 30:
                    score += 5
                    reasons.append("Expected within 30 days")
            except (ValueError, TypeError):
                pass  # Skip days scoring if conversion fails
            
            # 5. PURCHASE HISTORY
            try:
                purchase_count = int(float(row['purchase_count']))  # Handle float values
                if purchase_count >= 5:
                    score += 10
                    reasons.append("Established customer")
                elif purchase_count >= 3:
                    score += 5
                    reasons.append("Regular customer")
            except (ValueError, TypeError):
                pass  # Skip purchase count scoring if conversion fails
            
            # Categorize potential level
            if score >= 80:
                potential_level = "HOT LEAD"
                action = "Contact immediately"
            elif score >= 60:
                potential_level = "WARM LEAD" 
                action = "Contact this week"
            elif score >= 40:
                potential_level = "COLD LEAD"
                action = "Follow up next month"
            else:
                potential_level = "LOW POTENTIAL"
                action = "Monitor only"
            
            potential_buyers.append({
                'client': row['client'],
                'potential_score': score,
                'potential_level': potential_level,
                'action_required': action,
                'reasons': ', '.join(reasons) if reasons else "No specific reasons",
                'priority': row['priority'],
                'status': row['status'],
                'confidence': row['confidence'],
                'days_until_next': row['days_until_next'],
                'predicted_date': row['predicted_next_sale'],
                'last_purchase': row['last_invoice'],
                'purchase_count': row['purchase_count'],
                'buyer_type': row['buyer_type']
            })
        
        return pd.DataFrame(potential_buyers).sort_values('potential_score', ascending=False)
        
    def analyze_sales_potential(self, df):
        """Complete sales potential analysis"""
        # Get predictions first
        predictions, message = self.predict_next_sale(df)
        
        if predictions is None:
            return None, message
        
        # Identify potential buyers
        potential_df = self.identify_potential_buyers(predictions)
        
        return potential_df, "Potential buyer analysis completed"
    
    def calculate_confidence(self, client, predicted_gap):
        """Calculate prediction confidence based on client history"""
        if client not in self.client_history:
            return 0.5  # Default confidence for new clients
        
        client_data = self.client_history[client]
        
        # Base confidence on purchase history consistency
        if client_data['purchase_count'] < 3:
            return 0.6  # Low confidence for few purchases
        
        # Check if predicted gap is within historical range
        avg_gap = client_data['avg_gap']
        recent_gaps = client_data['recent_gaps']
        std_gap = np.std(recent_gaps) if len(recent_gaps) > 1 else avg_gap * 0.5
        
        # Confidence decreases as prediction deviates from historical average
        deviation = abs(predicted_gap - avg_gap) / (std_gap + 1)  # +1 to avoid division by zero
        confidence = max(0.3, 1 - deviation * 0.2)  # Minimum 30% confidence
        
        return min(0.95, confidence)  # Maximum 95% confidence
    
    def save_model(self, filepath='sales_model.pkl'):
        """Save the trained model and feature information"""
        try:
            model_data = {
                'model': self.model,
                'feature_columns': self.feature_columns,
                'model_performance': self.model_performance,
                'client_history': self.client_history
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            return True, "Model saved successfully"
        except Exception as e:
            return False, f"Failed to save model: {str(e)}"
    
    def load_model(self, filepath='sales_model.pkl'):
        """Load a previously trained model"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.feature_columns = model_data['feature_columns']
            self.model_performance = model_data['model_performance']
            self.client_history = model_data['client_history']
            
            return True, "Model loaded successfully"
        except Exception as e:
            return False, f"Failed to load model: {str(e)}"


# PyQt5 GUI Components
class PredictionWorker(QThread):
    finished = pyqtSignal(object, str)
    
    def __init__(self, predictor, df, mode='predict'):
        super().__init__()
        self.predictor = predictor
        self.df = df
        self.mode = mode
        
    def run(self):
        try:
            if self.mode == 'train':
                success, message = self.predictor.train_model(self.df)
                self.finished.emit(success, message)
            elif self.mode == 'predict':
                predictions, message = self.predictor.predict_next_sale(self.df)
                self.finished.emit(predictions, message)
        except Exception as e:
            self.finished.emit(None, str(e))

class PotentialAnalysisWorker(QThread):
    finished = pyqtSignal(object, str)
    
    def __init__(self, predictor, df):
        super().__init__()
        self.predictor = predictor
        self.df = df
        
    def run(self):
        try:
            potential_df, message = self.predictor.analyze_sales_potential(self.df)
            self.finished.emit(potential_df, message)
        except Exception as e:
            self.finished.emit(None, str(e))

class FrequentBuyersWorker(QThread):
    finished = pyqtSignal(object, str)
    
    def __init__(self, predictor, df):
        super().__init__()
        self.predictor = predictor
        self.df = df
        
    def run(self):
        try:
            frequent_buyers = self.predictor.identify_frequent_buyers(self.df)
            self.finished.emit(frequent_buyers, "Frequent buyer analysis completed")
        except Exception as e:
            self.finished.emit(None, str(e))

class SalesPredictionWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.predictor = SalesPredictionEngine()
        self.setup_ui()
        
        
    def setup_ui(self):
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("Sales Prediction Engine")
        title.setStyleSheet("font-size: 16pt; font-weight: bold; color: #e60540;")
        layout.addWidget(title)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.train_btn = QPushButton("Train Model")
        self.predict_btn = QPushButton("Predict Next Sales")
        self.potential_btn = QPushButton("Analyze Sales Potential")
        self.frequent_btn = QPushButton("Show Frequent Buyers")
        self.load_btn = QPushButton("Load Model")
        self.save_btn = QPushButton("Save Model")
        
        # Style buttons
        buttons = [self.train_btn, self.predict_btn, self.potential_btn, 
                  self.frequent_btn, self.load_btn, self.save_btn]
        
        for btn in buttons:
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #e60540;
                    color: white;
                    border: none;
                    border-radius: 5px;
                    padding: 8px 16px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #ff1e57;
                }
                QPushButton:disabled {
                    background-color: #cccccc;
                    color: #666666;
                }
            """)
        
        # Special styling for potential analysis button
        self.potential_btn.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #34ce57;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        
        self.train_btn.clicked.connect(self.train_model)
        self.predict_btn.clicked.connect(self.predict_sales)
        self.potential_btn.clicked.connect(self.analyze_potential)
        self.frequent_btn.clicked.connect(self.show_frequent_buyers)
        self.load_btn.clicked.connect(self.load_model)
        self.save_btn.clicked.connect(self.save_model)
        
        button_layout.addWidget(self.train_btn)
        button_layout.addWidget(self.predict_btn)
        button_layout.addWidget(self.potential_btn)
        button_layout.addWidget(self.frequent_btn)
        button_layout.addWidget(self.load_btn)
        button_layout.addWidget(self.save_btn)
        
        layout.addLayout(button_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #e60540;
                border-radius: 5px;
                text-align: center;
                color: white;
            }
            QProgressBar::chunk {
                background-color: #e60540;
            }
        """)
        layout.addWidget(self.progress_bar)
        
        # Results table
        self.results_table = QTableWidget()
        self.results_table.setStyleSheet("""
            QTableWidget {
                background-color: white;
                alternate-background-color: #f5f5f5;
                gridline-color: #ddd;
                color: black;
                font-size: 10pt;
            }
            QTableWidget::item {
                padding: 8px;
                border-bottom: 1px solid #eee;
            }
            QTableWidget::item:selected {
                background-color: #e60540;
                color: white;
            }
            QHeaderView::section {
                background-color: #e60540;
                color: white;
                padding: 10px;
                border: none;
                font-weight: bold;
                font-size: 10pt;
            }
        """)
        layout.addWidget(self.results_table)
        
        # Status display
        self.status_display = QTextEdit()
        self.status_display.setMaximumHeight(120)
        self.status_display.setReadOnly(True)
        self.status_display.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e2e;
                color: #ffffff;
                border: 1px solid #444;
                border-radius: 5px;
                padding: 8px;
                font-family: 'Courier New', monospace;
                font-size: 9pt;
            }
        """)
        layout.addWidget(self.status_display)
        
        self.setLayout(layout)
        self.setWindowTitle("Sales Prediction & Potential Analysis")
        self.resize(1200, 700)
        
    def log_status(self, message):
        timestamp = datetime.now().strftime('%H:%M:%S')
        self.status_display.append(f"[{timestamp}] {message}")
        
    def train_model(self):
        """Train model with data from CSV"""
        try:
            # Try different possible CSV files
            csv_files = ["AdminLT  Clients Data.csv", "DEMO.csv", "sales_data.csv"]
            df = None
            
            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file)
                    self.log_status(f"Loaded data from {csv_file}")
                    break
                except FileNotFoundError:
                    continue
            
            if df is None:
                QMessageBox.critical(self, "Error", "No CSV file found. Please check file names.")
                return
            
            self.worker = PredictionWorker(self.predictor, df, 'train')
            self.worker.finished.connect(self.on_training_complete)
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)
            self.set_buttons_enabled(False)
            self.worker.start()
            self.log_status("Training model started...")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load data: {str(e)}")
            self.log_status(f"Error: {str(e)}")
            
    def on_training_complete(self, success, message):
        self.progress_bar.setVisible(False)
        self.set_buttons_enabled(True)
        
        if success:
            QMessageBox.information(self, "Success", message)
            self.log_status(f"✓ Training completed: {message}")
            # Display performance metrics
            perf = self.predictor.model_performance
            self.log_status(f"📊 Model Performance:")
            self.log_status(f"   • MAE: {perf.get('mae', 'N/A'):.2f} days")
            self.log_status(f"   • R² Score: {perf.get('test_r2', 'N/A'):.3f}")
            self.log_status(f"   • Training samples: {perf.get('training_samples', 'N/A')}")
            self.log_status(f"   • Clients: {perf.get('total_clients', 'N/A')}")
        else:
            QMessageBox.warning(self, "Training Failed", message)
            self.log_status(f"✗ Training failed: {message}")
            
    def predict_sales(self):
        """Predict next sales"""
        try:
            # Try different possible CSV files
            csv_files = ["AdminLT  Clients Data.csv", "DEMO.csv", "sales_data.csv"]
            df = None
            
            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file)
                    self.log_status(f"Loaded data from {csv_file} for prediction")
                    break
                except FileNotFoundError:
                    continue
            
            if df is None:
                QMessageBox.critical(self, "Error", "No CSV file found for prediction.")
                return
            
            self.worker = PredictionWorker(self.predictor, df, 'predict')
            self.worker.finished.connect(self.on_prediction_complete)
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)
            self.set_buttons_enabled(False)
            self.worker.start()
            self.log_status("Generating predictions...")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load data: {str(e)}")
            self.log_status(f"Error: {str(e)}")
            
    def on_prediction_complete(self, predictions, message):
        self.progress_bar.setVisible(False)
        self.set_buttons_enabled(True)
        
        if predictions is not None:
            self.display_predictions(predictions)
            self.log_status(f"✓ Predictions generated: {message}")
            self.log_status(f"📈 Generated {len(predictions)} client predictions")
            
            # Show frequent buyer stats
            frequent_count = len(predictions[predictions['priority'].isin(['HIGH', 'MEDIUM'])])
            self.log_status(f"⭐ Frequent buyers: {frequent_count} clients")
            
        else:
            QMessageBox.warning(self, "Prediction Failed", message)
            self.log_status(f"✗ Prediction failed: {message}")
            
    def analyze_potential(self):
        """Analyze sales potential for all clients"""
        try:
            csv_files = ["AdminLT  Clients Data.csv", "DEMO.csv", "sales_data.csv"]
            df = None
            
            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file)
                    self.log_status(f"Loaded data from {csv_file} for potential analysis")
                    break
                except FileNotFoundError:
                    continue
            
            if df is None:
                QMessageBox.critical(self, "Error", "No CSV file found.")
                return
            
            self.worker = PotentialAnalysisWorker(self.predictor, df)
            self.worker.finished.connect(self.on_potential_analysis_complete)
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)
            self.set_buttons_enabled(False)
            self.worker.start()
            self.log_status("🔍 Analyzing sales potential...")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to analyze potential: {str(e)}")
            self.log_status(f"Error: {str(e)}")

    def on_potential_analysis_complete(self, potential_df, message):
        self.progress_bar.setVisible(False)
        self.set_buttons_enabled(True)
        
        if potential_df is not None:
            self.display_potential_analysis(potential_df)
            self.log_status(f"✅ {message}")
            
            # Show summary statistics
            hot_count = len(potential_df[potential_df['potential_level'] == 'HOT LEAD'])
            warm_count = len(potential_df[potential_df['potential_level'] == 'WARM LEAD'])
            cold_count = len(potential_df[potential_df['potential_level'] == 'COLD LEAD'])
            low_count = len(potential_df[potential_df['potential_level'] == 'LOW POTENTIAL'])
            
            self.log_status(f"🔥 Hot leads: {hot_count} clients (Contact immediately)")
            self.log_status(f"🌤️ Warm leads: {warm_count} clients (Contact this week)") 
            self.log_status(f"❄️ Cold leads: {cold_count} clients (Follow up next month)")
            self.log_status(f"📊 Low potential: {low_count} clients (Monitor only)")
            
        else:
            QMessageBox.warning(self, "Analysis Failed", message)
            self.log_status(f"❌ {message}")


    def safe_display_frequent_buyers(self, frequent_buyers_df):
        """Safe method to display frequent buyers data"""
        try:
            print("🛡️ SAFE DISPLAY METHOD STARTED")
            
            # Clear table
            self.results_table.clear()
            self.results_table.setRowCount(0)
            self.results_table.setColumnCount(0)
            
            if frequent_buyers_df is None or frequent_buyers_df.empty:
                print("No data to display")
                return
            
            # Convert to list with integer indices
            data_list = []
            for i, (client_name, row) in enumerate(frequent_buyers_df.iterrows()):
                data_list.append({
                    'row_index': i,  # This is ALWAYS an integer
                    'client_name': str(client_name),
                    'total_purchases': str(row.get('total_purchases', '')),
                    'avg_gap': str(row.get('avg_gap', '')),
                    'purchase_frequency': str(row.get('purchase_frequency', '')),
                    'buyer_type': str(row.get('buyer_type', '')),
                    'priority': str(row.get('priority', '')),
                    'days_since_last': str(row.get('days_since_last', '')),
                    'first_purchase': self.safe_format_date(row.get('first_purchase', '')),
                    'last_purchase': self.safe_format_date(row.get('last_purchase', ''))
                })
            
            print(f"🛡️ Converted {len(data_list)} records with integer indices")
            
            # Set up table headers
            headers = [
                'Client Name', 'Total Purchases', 'Avg Gap (days)', 
                'Purchase Frequency', 'Buyer Type', 'Priority',
                'Days Since Last', 'First Purchase', 'Last Purchase'
            ]
            self.results_table.setColumnCount(len(headers))
            self.results_table.setHorizontalHeaderLabels(headers)
            self.results_table.setRowCount(len(data_list))
            
            # Populate table - using ONLY the integer row_index from our list
            for data in data_list:
                row_index = data['row_index']  # This is guaranteed to be integer
                
                # Double convert to ensure integer
                safe_row_index = int(row_index)
                
                print(f"🛡️ Adding row {safe_row_index} with client: {data['client_name']}")
                
                # Add all columns for this row
                self.results_table.setItem(safe_row_index, 0, QTableWidgetItem(data['client_name']))
                self.results_table.setItem(safe_row_index, 1, QTableWidgetItem(data['total_purchases']))
                self.results_table.setItem(safe_row_index, 2, QTableWidgetItem(data['avg_gap']))
                self.results_table.setItem(safe_row_index, 3, QTableWidgetItem(data['purchase_frequency']))
                self.results_table.setItem(safe_row_index, 4, QTableWidgetItem(data['buyer_type']))
                self.results_table.setItem(safe_row_index, 5, QTableWidgetItem(data['priority']))
                self.results_table.setItem(safe_row_index, 6, QTableWidgetItem(data['days_since_last']))
                self.results_table.setItem(safe_row_index, 7, QTableWidgetItem(data['first_purchase']))
                self.results_table.setItem(safe_row_index, 8, QTableWidgetItem(data['last_purchase']))
            
            self.results_table.resizeColumnsToContents()
            print("✅ SAFE DISPLAY METHOD COMPLETED SUCCESSFULLY")
            
        except Exception as e:
            print(f"❌ SAFE DISPLAY METHOD FAILED: {str(e)}")
            import traceback
            traceback.print_exc()

    def safe_format_date(self, date_value):
        """Safely format date values without exceptions"""
        try:
            if hasattr(date_value, 'strftime'):
                return date_value.strftime("%Y-%m-%d")
            elif isinstance(date_value, str):
                return date_value
            else:
                return str(date_value)
        except:
            return str(date_value)
    
    def show_frequent_buyers(self):
        """Show analysis of frequent buyers - FIXED VERSION"""
        try:
            csv_files = ["AdminLT  Clients Data.csv", "DEMO.csv", "sales_data.csv"]
            df = None
            
            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file)
                    self.log_status(f"📊 Loaded data from {csv_file} for frequent buyers analysis")
                    break
                except FileNotFoundError:
                    continue
            
            if df is None:
                QMessageBox.critical(self, "Error", "No CSV file found.")
                return
            
            # SHOW PROGRESS BAR AND DISABLE BUTTONS
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)  # Indeterminate progress
            self.set_buttons_enabled(False)
            
            self.worker = FrequentBuyersWorker(self.predictor, df)
            self.worker.finished.connect(self.on_frequent_buyers_complete)
            self.worker.start()
            self.log_status("🔍 Analyzing frequent buyers patterns...")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to analyze frequent buyers: {str(e)}")
            # RE-ENABLE BUTTONS ON ERROR
            self.set_buttons_enabled(True)
            self.progress_bar.setVisible(False)

    def on_frequent_buyers_complete(self, frequent_buyers):
        """Handle completion of frequent buyers analysis - FIXED VERSION"""
        try:
            print(f"DEBUG: on_frequent_buyers_complete started")
            
            # HIDE PROGRESS BAR AND RE-ENABLE BUTTONS FIRST
            self.progress_bar.setVisible(False)
            self.set_buttons_enabled(True)
            
            if frequent_buyers is not None and not frequent_buyers.empty:
                print(f"DEBUG: Data shape: {frequent_buyers.shape}")
                
                # Display the data
                self.safe_display_frequent_buyers(frequent_buyers)
                
                # Update status
                status_msg = f"Found {len(frequent_buyers)} frequent buyers"
                self.log_status(f"✅ {status_msg}")
                
                # Show summary statistics
                high_priority = len(frequent_buyers[frequent_buyers['priority'] == 'HIGH'])
                medium_priority = len(frequent_buyers[frequent_buyers['priority'] == 'MEDIUM'])
                low_priority = len(frequent_buyers[frequent_buyers['priority'] == 'LOW'])
                
                self.log_status(f"🎯 Priority Summary:")
                self.log_status(f"   • HIGH: {high_priority} clients")
                self.log_status(f"   • MEDIUM: {medium_priority} clients")
                self.log_status(f"   • LOW: {low_priority} clients")
                
            else:
                status_msg = "No frequent buyers found"
                self.log_status(f"❌ {status_msg}")
                self.results_table.setRowCount(0)
                
        except Exception as e:
            error_msg = f"Error displaying frequent buyers: {str(e)}"
            print(error_msg)
            self.log_status(f"❌ {error_msg}")
            # Ensure buttons are re-enabled even on error
            self.set_buttons_enabled(True)
            self.progress_bar.setVisible(False)

    def start_frequent_buyers_analysis(self):
        """Start the frequent buyers analysis"""
        try:
            # SHOW PROGRESS BAR
            if hasattr(self, 'progress_bar'):
                self.progress_bar.show()
                self.progress_bar.setRange(0, 0)  # Indeterminate progress
                
            if hasattr(self, 'progress_label'):
                self.progress_label.show()
                self.progress_label.setText("Analyzing frequent buyers...")
                
            # Update status
            if hasattr(self, 'status_label'):
                self.status_label.setText("Analyzing purchase patterns...")
                
            # Start analysis in thread
            self.analysis_thread = FrequentBuyersThread(self.sales_engine, self.df)
            self.analysis_thread.finished.connect(self.on_frequent_buyers_complete)
            self.analysis_thread.start()
            
        except Exception as e:
            print(f"Error starting analysis: {str(e)}")
            
            # Hide progress bar on error
            if hasattr(self, 'progress_bar'):
                self.progress_bar.hide()
            if hasattr(self, 'progress_label'):
                self.progress_label.hide()
    
    def display_frequent_buyers(self, frequent_buyers_df):
        """Completely bulletproof display method"""
        try:
            print(f"DEBUG: Starting display_frequent_buyers")
            print(f"DEBUG: Input type: {type(frequent_buyers_df)}")
            
            # Clear the table first
            self.results_table.setRowCount(0)
            
            if frequent_buyers_df is None or frequent_buyers_df.empty:
                print("DEBUG: No data to display")
                return
                
            print(f"DEBUG: Data shape: {frequent_buyers_df.shape}")
            
            # Set column headers
            headers = [
                'Client Name', 'Total Purchases', 'Avg Gap (days)', 
                'Purchase Frequency', 'Buyer Type', 'Priority',
                'Days Since Last', 'First Purchase', 'Last Purchase'
            ]
            self.results_table.setColumnCount(len(headers))
            self.results_table.setHorizontalHeaderLabels(headers)
            
            # Convert DataFrame to a simple list of dictionaries with reset index
            df_reset = frequent_buyers_df.reset_index()
            records = df_reset.to_dict('records')
            
            print(f"DEBUG: Number of records to display: {len(records)}")
            
            # Set the row count
            self.results_table.setRowCount(len(records))
            
            # Use simple integer loop - THIS CANNOT FAIL
            for i in range(len(records)):
                # i is ALWAYS an integer
                row_data = records[i]
                
                # Extract values with defaults
                client_name = str(row_data.get('client name', row_data.get('index', f'Client_{i}')))
                total_purchases = str(row_data.get('total_purchases', ''))
                avg_gap = str(row_data.get('avg_gap', ''))
                purchase_frequency = str(row_data.get('purchase_frequency', ''))
                buyer_type = str(row_data.get('buyer_type', ''))
                priority = str(row_data.get('priority', ''))
                days_since_last = str(row_data.get('days_since_last', ''))
                
                # Format dates safely
                first_purchase = self.safe_format_date(row_data.get('first_purchase', ''))
                last_purchase = self.safe_format_date(row_data.get('last_purchase', ''))
                
                # Create all items for this row
                items = [
                    QTableWidgetItem(client_name),
                    QTableWidgetItem(total_purchases),
                    QTableWidgetItem(avg_gap),
                    QTableWidgetItem(purchase_frequency),
                    QTableWidgetItem(buyer_type),
                    QTableWidgetItem(priority),
                    QTableWidgetItem(days_since_last),
                    QTableWidgetItem(first_purchase),
                    QTableWidgetItem(last_purchase)
                ]
                
                # Add all items to the table - i is guaranteed to be integer
                for col_idx, item in enumerate(items):
                    # Double-check that i is integer
                    int_row_idx = int(i)
                    self.results_table.setItem(int_row_idx, col_idx, item)
                    
                if i % 50 == 0:  # Progress indicator for large datasets
                    print(f"DEBUG: Processed {i} rows...")
            
            print("DEBUG: Table population completed successfully")
            self.results_table.resizeColumnsToContents()
            
        except Exception as e:
            print(f"ERROR in display_frequent_buyers: {str(e)}")
            import traceback
            traceback.print_exc()

    def safe_format_date(self, date_value):
        """Safely format date values without exceptions"""
        try:
            if hasattr(date_value, 'strftime'):
                return date_value.strftime("%Y-%m-%d")
            elif isinstance(date_value, str):
                return date_value
            else:
                return str(date_value)
        except:
            return str(date_value)    
    def add_table_row(self, row_idx, client_name, row_data):
        """Add a single row to the table - row_idx is guaranteed to be integer"""
        try:
            # Client Name (column 0)
            item = QTableWidgetItem(str(client_name))
            self.results_table.setItem(row_idx, 0, item)
            
            # Total Purchases (column 1)
            item = QTableWidgetItem(str(row_data.get('total_purchases', '')))
            self.results_table.setItem(row_idx, 1, item)
            
            # Avg Gap (column 2)
            item = QTableWidgetItem(str(row_data.get('avg_gap', '')))
            self.results_table.setItem(row_idx, 2, item)
            
            # Purchase Frequency (column 3)
            item = QTableWidgetItem(str(row_data.get('purchase_frequency', '')))
            self.results_table.setItem(row_idx, 3, item)
            
            # Buyer Type (column 4)
            item = QTableWidgetItem(str(row_data.get('buyer_type', '')))
            self.results_table.setItem(row_idx, 4, item)
            
            # Priority (column 5)
            item = QTableWidgetItem(str(row_data.get('priority', '')))
            self.results_table.setItem(row_idx, 5, item)
            
            # Days Since Last (column 6)
            item = QTableWidgetItem(str(row_data.get('days_since_last', '')))
            self.results_table.setItem(row_idx, 6, item)
            
            # First Purchase (column 7)
            first_purchase = row_data.get('first_purchase', '')
            if hasattr(first_purchase, 'strftime'):
                first_purchase = first_purchase.strftime("%Y-%m-%d")
            item = QTableWidgetItem(str(first_purchase))
            self.results_table.setItem(row_idx, 7, item)
            
            # Last Purchase (column 8)
            last_purchase = row_data.get('last_purchase', '')
            if hasattr(last_purchase, 'strftime'):
                last_purchase = last_purchase.strftime("%Y-%m-%d")
            item = QTableWidgetItem(str(last_purchase))
            self.results_table.setItem(row_idx, 8, item)
            
        except Exception as e:
            print(f"Error adding row {row_idx}: {str(e)}")
    def display_predictions(self, predictions_df):
        self.results_table.setRowCount(len(predictions_df))
        self.results_table.setColumnCount(len(predictions_df.columns))
        self.results_table.setHorizontalHeaderLabels(predictions_df.columns)
        
        for row_idx, row_data in predictions_df.iterrows():
            for col_idx, value in enumerate(row_data):
                # Convert value to string safely
                if pd.isna(value):
                    display_value = ""
                else:
                    display_value = str(value)
                
                item = QTableWidgetItem(display_value)
                item.setForeground(QBrush(QColor("black")))
                
                # Color code based on priority
                if 'priority' in predictions_df.columns and col_idx == predictions_df.columns.get_loc('priority'):
                    if value == 'HIGH':
                        item.setBackground(QColor("#ffcccc"))
                        item.setForeground(QBrush(QColor("darkred")))
                    elif value == 'MEDIUM':
                        item.setBackground(QColor("#fff4cc"))
                        item.setForeground(QBrush(QColor("darkorange")))
                    elif value == 'LOW':
                        item.setBackground(QColor("#e6f3ff"))
                        item.setForeground(QBrush(QColor("darkblue")))
                
                # Color code days until next
                elif 'days_until_next' in predictions_df.columns and col_idx == predictions_df.columns.get_loc('days_until_next'):
                    try:
                        days = int(float(value))  # Handle float values
                        if days <= 7:
                            item.setBackground(QColor("#ffcccc"))
                            item.setForeground(QBrush(QColor("darkred")))
                        elif days <= 14:
                            item.setBackground(QColor("#fff4cc"))
                            item.setForeground(QBrush(QColor("darkorange")))
                        elif days <= 30:
                            item.setBackground(QColor("#ccffcc"))
                            item.setForeground(QBrush(QColor("darkgreen")))
                    except (ValueError, TypeError):
                        pass  # Skip coloring if conversion fails
                
                # Color code status
                elif 'status' in predictions_df.columns and col_idx == predictions_df.columns.get_loc('status'):
                    if 'OVERDUE' in str(value):
                        item.setBackground(QColor("#ffcccc"))
                        item.setForeground(QBrush(QColor("darkred")))
                    elif 'ON_TRACK' in str(value):
                        item.setBackground(QColor("#ccffcc"))
                        item.setForeground(QBrush(QColor("darkgreen")))
                
                self.results_table.setItem(row_idx, col_idx, item)
                
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.results_table.resizeRowsToContents()        
    
    def display_potential_analysis(self, potential_df):
        """Display potential buyer analysis with color coding"""
        self.results_table.setRowCount(len(potential_df))
        self.results_table.setColumnCount(len(potential_df.columns))
        self.results_table.setHorizontalHeaderLabels(potential_df.columns)
        
        for row_idx, row_data in potential_df.iterrows():
            for col_idx, value in enumerate(row_data):
                item = QTableWidgetItem(str(value))
                item.setForeground(QBrush(QColor("black")))
                
                # Color code by potential level
                if 'potential_level' in potential_df.columns and col_idx == potential_df.columns.get_loc('potential_level'):
                    if value == 'HOT LEAD':
                        item.setBackground(QColor("#ff6b6b"))  # Red
                        item.setForeground(QBrush(QColor("white")))
                    elif value == 'WARM LEAD':
                        item.setBackground(QColor("#ffd93d"))  # Yellow
                        item.setForeground(QBrush(QColor("black")))
                    elif value == 'COLD LEAD':
                        item.setBackground(QColor("#6bc5d2"))  # Blue
                        item.setForeground(QBrush(QColor("white")))
                    elif value == 'LOW POTENTIAL':
                        item.setBackground(QColor("#95a5a6"))  # Gray
                        item.setForeground(QBrush(QColor("white")))
                
                # Color code action required
                elif 'action_required' in potential_df.columns and col_idx == potential_df.columns.get_loc('action_required'):
                    if 'immediately' in str(value).lower():
                        item.setBackground(QColor("#ff6b6b"))  # Red
                        item.setForeground(QBrush(QColor("white")))
                    elif 'this week' in str(value).lower():
                        item.setBackground(QColor("#ffd93d"))  # Yellow
                        item.setForeground(QBrush(QColor("black")))
                    elif 'next month' in str(value).lower():
                        item.setBackground(QColor("#6bc5d2"))  # Blue
                        item.setForeground(QBrush(QColor("white")))
                
                # Color code priority
                elif 'priority' in potential_df.columns and col_idx == potential_df.columns.get_loc('priority'):
                    if value == 'HIGH':
                        item.setBackground(QColor("#ff6b6b"))
                        item.setForeground(QBrush(QColor("white")))
                    elif value == 'MEDIUM':
                        item.setBackground(QColor("#ffd93d"))
                        item.setForeground(QBrush(QColor("black")))
                    elif value == 'LOW':
                        item.setBackground(QColor("#6bc5d2"))
                        item.setForeground(QBrush(QColor("white")))
            
                self.results_table.setItem(row_idx, col_idx, item)
    
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.results_table.resizeRowsToContents()

            
    def load_model(self):
        success, message = self.predictor.load_model()
        if success:
            QMessageBox.information(self, "Success", message)
            self.log_status(f"✓ {message}")
            perf = self.predictor.model_performance
            if perf:
                self.log_status(f"📊 Loaded Model Info:")
                self.log_status(f"   • Last trained: {perf.get('last_trained', 'N/A')}")
                self.log_status(f"   • Training samples: {perf.get('training_samples', 'N/A')}")
                self.log_status(f"   • MAE: {perf.get('mae', 'N/A'):.2f} days")
        else:
            QMessageBox.warning(self, "Load Failed", message)
            self.log_status(f"✗ {message}")
            
    def save_model(self):
        success, message = self.predictor.save_model()
        if success:
            QMessageBox.information(self, "Success", message)
            self.log_status(f"✓ {message}")
        else:
            QMessageBox.warning(self, "Save Failed", message)
            self.log_status(f"✗ {message}")
            
    def set_buttons_enabled(self, enabled):
        """Enable/disable all buttons during operations"""
        self.train_btn.setEnabled(enabled)
        self.predict_btn.setEnabled(enabled)
        self.potential_btn.setEnabled(enabled)
        self.frequent_btn.setEnabled(enabled)
        self.load_btn.setEnabled(enabled)
        self.save_btn.setEnabled(enabled)


# Main application
def main():
    app = QApplication(sys.argv)
    
    # Set dark theme for the main window
    app.setStyleSheet("""
        QWidget {
            background-color: #181824;
            color: white;
            font-family: Arial;
        }
        QLabel {
            color: white;
        }
    """)
    
    window = SalesPredictionWidget()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()