import streamlit as st # Import Streamlit for web app
import pandas as pd # Import Pandas for data manipulation
import numpy as np # Import NumPy for numerical operations
import matplotlib.pyplot as plt # Import Matplotlib for plotting graphs 
import seaborn as sns # Import Seaborn for enhanced visualizations 
from sklearn.model_selection import train_test_split # Import train_test_split for splitting data into training and testing sets
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor # Import Random Forest models for classification and regression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score,
    confusion_matrix, roc_curve, auc
) # Import various metrics for model evaluation
from sklearn.preprocessing import LabelEncoder, OneHotEncoder # Import encoders for categorical variables
from sklearn.compose import ColumnTransformer  # Import ColumnTransformer for preprocessing
from sklearn.pipeline import Pipeline # Import Pipeline for creating a machine learning pipeline
import pickle # Import pickle for saving the model
import io # Import io for handling byte streams
import re # Import re for regular expressions

# Domain keywords dictionary
DOMAIN_KEYWORDS = {
    'healthcare': [
        'patient', 'diagnosis', 'treatment', 'disease', 'symptom', 'medical',
        'health', 'hospital', 'doctor', 'blood', 'pressure', 'heart', 'cancer',
        'diabetes', 'bmi', 'age', 'gender', 'weight', 'height'
    ],
    'finance': [
        'price', 'cost', 'revenue', 'profit', 'loss', 'income', 'expense',
        'balance', 'account', 'transaction', 'stock', 'market', 'investment',
        'interest', 'rate', 'loan', 'credit', 'debit', 'bank'
    ],
    'education': [
        'student', 'grade', 'score', 'exam', 'test', 'course', 'class',
        'school', 'university', 'college', 'education', 'learning', 'study',
        'attendance', 'performance', 'teacher', 'subject'
    ],
    'retail': [
        'product', 'customer', 'sale', 'purchase', 'order', 'item', 'price',
        'quantity', 'store', 'shop', 'retail', 'inventory', 'stock', 'discount',
        'category', 'brand'
    ],
    'agriculture': [
        'crop', 'yield', 'soil', 'temperature', 'rainfall', 'fertilizer',
        'farm', 'harvest', 'plant', 'agriculture', 'pesticide', 'irrigation',
        'humidity', 'moisture', 'climate'
    ],
    'transportation': [
        'vehicle', 'speed', 'traffic', 'route', 'distance', 'fuel', 'driver',
        'trip', 'accident', 'transport', 'road', 'logistics', 'shipment', 'travel'
    ],
    'environment': [
        'pollution', 'air', 'water', 'emission', 'carbon', 'climate', 'temperature',
        'weather', 'ozone', 'recycle', 'waste', 'green', 'energy', 'ecology'
    ],
    'sports': [
        'match', 'team', 'player', 'score', 'goal', 'tournament', 'win', 'lose',
        'stadium', 'coach', 'league', 'game', 'athlete', 'performance'
    ],
    'ecommerce': [
        'user', 'cart', 'checkout', 'payment', 'wishlist', 'delivery', 'return',
        'review', 'rating', 'browse', 'recommendation', 'seller', 'shipping'
    ],
    'real_estate': [
        'property', 'house', 'apartment', 'rent', 'buy', 'sell', 'price', 'location',
        'area', 'bedroom', 'bathroom', 'agent', 'mortgage', 'listing'
    ],
    'employment': [
        'job', 'employee', 'employer', 'salary', 'position', 'department',
        'resume', 'interview', 'hiring', 'contract', 'benefits', 'experience'
    ],
    'energy': [
        'electricity', 'power', 'solar', 'wind', 'generation', 'consumption',
        'renewable', 'grid', 'voltage', 'current', 'energy', 'unit', 'supply'
    ],
    'technology': [
        'device', 'software', 'hardware', 'update', 'bug', 'version', 'release',
        'system', 'application', 'performance', 'technology', 'tool', 'network'
    ],
    'tourism': [
        'destination', 'tourist', 'package', 'hotel', 'travel', 'flight', 'booking',
        'sightseeing', 'guide', 'trip', 'location', 'visa', 'itinerary'
    ]
}

# Set page config
st.set_page_config(
    page_title="ML Model Trainer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS 
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .css-1d391kg {
        padding: 2rem 1rem;
    }
    .domain-info {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True) 

class MLApp:
    def __init__(self): # Initialize the app with default values
        self.data = None
        self.X = None
        self.y = None
        self.model = None
        self.is_classification = None
        self.domain = None
        self.domain_confidence = None
        self.preprocessor = None
        self.target_column = None
        self.feature_columns = None
        self.categorical_features = None
        self.numerical_features = None
        
    def detect_domain(self, column_names):
        """Detect the domain of the dataset based on column names."""
        column_text = ' '.join(column_names).lower()
        domain_scores = {}
        
        for domain, keywords in DOMAIN_KEYWORDS.items():
            score = sum(1 for keyword in keywords if keyword in column_text)
            if score > 0:
                domain_scores[domain] = score
        
        if domain_scores:
            # Get the domain with highest score
            max_domain = max(domain_scores.items(), key=lambda x: x[1]) #lambda function to get the item with maximum score
            # Calculate confidence as percentage of maximum possible score
            max_possible_score = len(DOMAIN_KEYWORDS[max_domain[0]])
            confidence = (max_domain[1] / max_possible_score) * 100
            return max_domain[0], confidence
        
        return "General", 0

    def detect_problem_type(self, target_series):
        """Detect if the problem is classification or regression."""
        unique_values = len(target_series.unique()) # length of unique values in target series
        is_categorical = target_series.dtype == 'object' or unique_values <= 10 # if the target series is categorical or has less than or equal to 10 unique values 
        
        if is_categorical:
            return "Classification", unique_values
        return "Regression", unique_values

    def perform_eda(self):
        """Perform Exploratory Data Analysis."""
        st.subheader("📊 Exploratory Data Analysis") # summarize its main characteristics

        # Create tabs for different EDA sections
        tab1, tab2, tab3, tab4 = st.tabs(["Data Overview", "Missing Values", "Value Counts", "Correlations"])
        
        with tab1:
            st.write("### Basic Information")
            
            # Create a summary table
            summary_data = {
                'Metric': [
                    'Number of Rows',
                    'Number of Columns',
                    'Memory Usage',
                    'Target Column',
                    'Problem Type',
                    'Number of Numerical Features',
                    'Number of Categorical Features' # i.e., features with object data type
                ],
                'Value': [
                    len(self.data),
                    len(self.data.columns),
                    f"{self.data.memory_usage(deep=True).sum() / 1024:.2f} KB",
                    self.target_column,
                    "Classification" if self.is_classification else "Regression",
                    len(self.data.select_dtypes(include=[np.number]).columns),
                    len(self.data.select_dtypes(include=['object']).columns)
                ]
            }
            
            # Display summary table
            st.table(pd.DataFrame(summary_data)) # Create a DataFrame from the summary data used for display

            # Display data types in a table
            st.write("### Data Types")
            dtype_data = pd.DataFrame({
                'Column': self.data.columns,
                'Data Type': self.data.dtypes,
                'Non-Null Count': self.data.count(),
                'Null Count': self.data.isnull().sum(),
                'Null Percentage': (self.data.isnull().sum() / len(self.data) * 100).round(2)
            })
            st.table(dtype_data)
            
            # Display statistical summary
            st.write("### Statistical Summary")
            st.write(self.data.describe()) # describe() provides a statistical summary of the DataFrame
            
            # Display first few rows
            st.write("### Data Preview")
            st.dataframe(self.data.head())
        
        with tab2:
            st.write("### Missing Values Analysis")
            missing_data = pd.DataFrame({
                'Missing Values': self.data.isnull().sum(), # Count of missing values in each column
                'Percentage': (self.data.isnull().sum() / len(self.data)) * 100 # Percentage of missing values in each column
            })
            st.write(missing_data[missing_data['Missing Values'] > 0]) # Display columns with missing values
            
            # Visualize missing values
            if missing_data['Missing Values'].sum() > 0:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.heatmap(self.data.isnull(), yticklabels=False, cbar=False, cmap='viridis', ax=ax)
                plt.title('Missing Values Heatmap')
                st.pyplot(fig)
                plt.close()
        
        with tab3:
            st.write("### Value Counts for Categorical Features")
            categorical_cols = self.data.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                st.write(f"#### {col}")
                value_counts = self.data[col].value_counts()
                st.write(value_counts)
                
                # Plot value counts
                fig, ax = plt.subplots(figsize=(10, 6))
                value_counts.plot(kind='bar', ax=ax)
                plt.title(f'Value Counts for {col}')
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
        
        with tab4:
            st.write("### Correlation Analysis")
            # Calculate correlations for numerical columns
            numerical_cols = self.data.select_dtypes(include=[np.number]).columns
            if len(numerical_cols) > 1:
                corr_matrix = self.data[numerical_cols].corr()
                
                # Plot correlation heatmap
                fig, ax = plt.subplots(figsize=(12, 8))
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax) 
                plt.title('Correlation Heatmap')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

    def load_data(self, uploaded_file):
        """Load and validate the CSV file."""
        try:
            self.data = pd.read_csv(uploaded_file)
            
            # Set target column (last column)
            self.target_column = self.data.columns[-1]
            self.feature_columns = self.data.columns[:-1]
            
            # Detect domain
            self.domain, self.domain_confidence = self.detect_domain(self.data.columns)
            
            # Basic data cleaning
            self.data = self.data.replace([np.inf, -np.inf], np.nan)
            
            # Drop columns with too many missing values (>50%)
            missing_threshold = len(self.data) * 0.5
            columns_to_drop = self.data.columns[self.data.isnull().sum() > missing_threshold]
            if len(columns_to_drop) > 0:
                self.data = self.data.drop(columns=columns_to_drop)
                st.warning(f"Dropped columns with >50% missing values: {', '.join(columns_to_drop)}")
            
            return True
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return False

    def preprocess_data(self):
        """Preprocess the data and prepare features."""
        try:
            # Identify categorical and numerical features
            self.categorical_features = self.data[self.feature_columns].select_dtypes(include=['object']).columns
            self.numerical_features = self.data[self.feature_columns].select_dtypes(include=[np.number]).columns
            
            # Create preprocessing pipeline
            categorical_transformer = Pipeline(steps=[
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])
            
            numerical_transformer = Pipeline(steps=[
                ('imputer', 'passthrough')  # We'll handle missing values in the data
            ])
            
            self.preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numerical_transformer, self.numerical_features),
                    ('cat', categorical_transformer, self.categorical_features)
                ])
            
            # Prepare features and target
            self.X = self.data[self.feature_columns]
            self.y = self.data[self.target_column]
            
            # Handle target variable encoding if categorical
            if self.y.dtype == 'object':
                self.label_encoder = LabelEncoder()
                self.y = self.label_encoder.fit_transform(self.y)
            
            # Determine if classification or regression
            problem_type, unique_values = self.detect_problem_type(self.data[self.target_column])
            self.is_classification = problem_type == "Classification"
            
            return True
        except Exception as e:
            st.error(f"Error preprocessing data: {str(e)}")
            return False

    def train_model(self):
        """Train the appropriate Random Forest model."""
        try:
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                self.X, self.y, test_size=0.2, random_state=42
            )

            # Create and train the model pipeline
            if self.is_classification:
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)

            # Create full pipeline
            self.model = Pipeline(steps=[
                ('preprocessor', self.preprocessor),
                ('classifier', model)
            ])

            # Train the model
            self.model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = self.model.predict(X_test)
            
            return X_test, y_test, y_pred
        except Exception as e:
            st.error(f"Error training model: {str(e)}")
            return None, None, None

    def display_metrics(self, y_true, y_pred):
        """Display appropriate metrics based on problem type."""
        st.subheader("📊 Model Performance Metrics")
        
        if self.is_classification:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.4f}")
                st.metric("Precision", f"{precision_score(y_true, y_pred, average='weighted'):.4f}")
            with col2:
                st.metric("Recall", f"{recall_score(y_true, y_pred, average='weighted'):.4f}")
                st.metric("F1 Score", f"{f1_score(y_true, y_pred, average='weighted'):.4f}")
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("MAE", f"{mean_absolute_error(y_true, y_pred):.4f}")
                st.metric("RMSE", f"{np.sqrt(mean_squared_error(y_true, y_pred)):.4f}")
            with col2:
                st.metric("R² Score", f"{r2_score(y_true, y_pred):.4f}")

    def generate_visualizations(self, X_test, y_true, y_pred):
        """Generate appropriate visualizations."""
        st.subheader("📈 Model Visualizations")
        
        # Get feature names after preprocessing
        try:
            # Get numerical feature names
            feature_names = list(self.numerical_features)
            
            # Get categorical feature names
            if len(self.categorical_features) > 0:
                ohe = self.preprocessor.named_transformers_['cat'].named_steps['onehot']
                for i, col in enumerate(self.categorical_features):
                    categories = ohe.categories_[i]
                    feature_names.extend([f"{col}_{cat}" for cat in categories])
            
            # Feature Importance Plot
            st.write("### Feature Importance")
            importances = pd.DataFrame({
                'Feature': feature_names,
                'Importance': self.model.named_steps['classifier'].feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=importances.head(10), x='Importance', y='Feature', ax=ax)
            plt.title('Top 10 Most Important Features')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        except Exception as e:
            st.warning(f"Could not generate feature importance plot: {str(e)}")
            # Fallback to basic feature importance
            importances = pd.DataFrame({
                'Feature': self.feature_columns,
                'Importance': self.model.named_steps['classifier'].feature_importances_[:len(self.feature_columns)]
            }).sort_values('Importance', ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=importances.head(10), x='Importance', y='Feature', ax=ax)
            plt.title('Top 10 Most Important Features')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        if self.is_classification:
            # Confusion Matrix
            st.write("### Confusion Matrix")
            cm = confusion_matrix(y_true, y_pred)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            st.pyplot(fig)
            plt.close()

            # ROC Curve (for binary classification)
            if len(np.unique(y_true)) == 2:
                st.write("### ROC Curve")
                fig, ax = plt.subplots(figsize=(8, 6))
                fpr, tpr, _ = roc_curve(y_true, y_pred)
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, color='darkorange', lw=2, 
                       label=f'ROC curve (AUC = {roc_auc:.2f})')
                ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title('Receiver Operating Characteristic (ROC) Curve')
                ax.legend(loc="lower right")
                st.pyplot(fig)
                plt.close()
        else:
            # Residual Plot
            st.write("### Residual Plot")
            residuals = y_true - y_pred
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.scatterplot(x=y_pred, y=residuals, ax=ax)
            ax.axhline(y=0, color='r', linestyle='--')
            ax.set_xlabel('Predicted Values')
            ax.set_ylabel('Residuals')
            ax.set_title('Residual Plot')
            st.pyplot(fig)
            plt.close()

    def save_model(self):
        """Save the trained model and preprocessing information."""
        try:
            model_data = {
                'model': self.model,
                'is_classification': self.is_classification,
                'feature_columns': self.feature_columns,
                'target_column': self.target_column,
                'domain': self.domain
            }
            buffer = io.BytesIO()
            pickle.dump(model_data, buffer)
            return buffer
        except Exception as e:
            st.error(f"Error saving model: {str(e)}")
            return None

    def display_domain_info(self):
        """Display domain information in the sidebar."""
        with st.sidebar:
            st.subheader("📊 Dataset Information")
            
            # Domain information
            domain_color = "green" if self.domain_confidence > 50 else "orange"
            st.markdown(f"""
                <div class="domain-info" style="background-color: {domain_color}20; border: 1px solid {domain_color}50;">
                    <h4>Detected Domain: {self.domain.title()}</h4>
                    <p>Confidence: {self.domain_confidence:.1f}%</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Problem type information
            problem_type = "Classification" if self.is_classification else "Regression"
            st.markdown(f"""
                <div class="domain-info" style="background-color: #0e1117; border: 1px solid #4dabf7;">
                    <h4>Problem Type: {problem_type}</h4>
                </div>
            """, unsafe_allow_html=True)

def main():
    st.title("🤖 Machine Learning Model Trainer")
    
    # Sidebar
    with st.sidebar:
        st.header("About")
        st.write("""
        This app helps you train machine learning models on your data:
        1. Upload your CSV file
        2. View data analysis
        3. Train model automatically
        4. View results and metrics
        """)
    
    # Main content
    st.write("Upload your CSV file to train a Random Forest model for classification or regression.")
    
    # Initialize the app
    app = MLApp()
    
    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
       if app.load_data(uploaded_file):
        if app.preprocess_data():  # Moved this up before domain info
            # Display domain information
            app.display_domain_info()
            
            # Perform EDA
            app.perform_eda()
            
            if st.button("Train Model"):
                with st.spinner("Training model..."):
                    X_test, y_test, y_pred = app.train_model()
                    
                    if X_test is not None:
                        # Display metrics and visualizations
                        app.display_metrics(y_test, y_pred)
                        app.generate_visualizations(X_test, y_test, y_pred)
                        
                        # Save model option
                        st.subheader("💾 Save Model")
                        if st.button("Download Trained Model"):
                            buffer = app.save_model()
                            if buffer:
                                st.download_button(
                                    label="Click to Download",
                                    data=buffer.getvalue(),
                                    file_name="trained_model.pkl",
                                    mime="application/octet-stream"
                                )


if __name__ == "__main__":
    main() 