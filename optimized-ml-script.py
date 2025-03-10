import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import shap
from joblib import dump, Parallel, delayed
import logging
import warnings
from functools import partial

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Set random seed for reproducibility
np.random.seed(42)

def load_and_prepare_data(file_path):
    """Load data and perform initial preparation."""
    logger.info(f"Loading data from {file_path}")
    try:
        data = pd.read_excel(file_path)
        logger.info(f"Dataset shape: {data.shape}")
        logger.info(f"Columns: {data.columns.tolist()}")
        
        # Check for missing values
        missing_values = data.isnull().sum()
        if missing_values.sum() > 0:
            logger.warning(f"Missing values detected:\n{missing_values[missing_values > 0]}")
        
        return data
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def preprocess_features(X):
    """Preprocess features including encoding categoricals."""
    # Handle categorical columns
    categorical_cols = X.select_dtypes(include=['object', 'bool']).columns
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    # Keep only numeric columns
    X = X.select_dtypes(include=['int64', 'float64'])
    
    return X

def evaluate_model(model, X_test, y_test, outcome_column, feature_names):
    """Evaluate model and generate visualizations."""
    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    clf_report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    
    # Try to calculate ROC-AUC
    try:
        roc_auc = roc_auc_score(y_test, y_prob)
    except Exception as e:
        logger.warning(f"ROC-AUC could not be calculated: {e}")
        roc_auc = None
    
    # Generate plots (we'll return the figure objects instead of showing them)
    figures = {}
    
    # Confusion matrix
    fig_cm = plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f'Confusion Matrix - {outcome_column}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    figures['confusion_matrix'] = fig_cm
    
    # ROC curve
    if roc_auc is not None:
        fig_roc = plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {outcome_column}')
        plt.legend()
        figures['roc_curve'] = fig_roc
    
    # Feature importance
    fig_imp = plt.figure(figsize=(12, 8))
    feature_importances = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    })
    top_features = feature_importances.sort_values('Importance', ascending=False).head(20)
    sns.barplot(x='Importance', y='Feature', data=top_features)
    plt.title(f'Top 20 Feature Importances - {outcome_column}')
    plt.tight_layout()
    figures['feature_importance'] = fig_imp
    
    # SHAP analysis (if sample size is sufficient)
    if X_test.shape[0] >= 1:
        try:
            # Create explainer and sample data for SHAP
            explainer = shap.TreeExplainer(model)
            shap_sample_size = min(20, X_test.shape[0])
            X_test_sample = X_test.iloc[:shap_sample_size]
            
            # Get SHAP values
            shap_values = explainer.shap_values(X_test_sample)
            
            # Handle SHAP values structure
            if isinstance(shap_values, list) and len(shap_values) == 2:
                shap_values_selected = shap_values[1]  # Use positive class SHAP values
                base_value = explainer.expected_value
                if isinstance(base_value, (list, np.ndarray)):
                    base_value = base_value[1] if len(base_value) > 1 else base_value[0]
            else:
                shap_values_selected = shap_values
                base_value = explainer.expected_value
                if isinstance(base_value, (list, np.ndarray)):
                    base_value = base_value[0]
            
            # Create SHAP waterfall plot for first instance
            instance_idx = 0
            shap_values_instance = shap_values_selected[instance_idx]
            
            # Ensure 1D for waterfall plot
            if len(np.shape(shap_values_instance)) > 1:
                shap_values_instance = shap_values_instance[:, 1]
            
            # Create SHAP Explanation object
            shap_exp = shap.Explanation(
                values=shap_values_instance,
                base_values=base_value,
                data=X_test_sample.iloc[instance_idx].values,
                feature_names=X_test_sample.columns.tolist()
            )
            
            # Create waterfall plot
            fig_shap = plt.figure(figsize=(12, 8))
            shap.waterfall_plot(shap_exp)
            plt.title(f'SHAP Waterfall Plot (First Test Sample) - {outcome_column}')
            plt.tight_layout()
            figures['shap_waterfall'] = fig_shap
            
        except Exception as e:
            logger.warning(f"SHAP analysis error: {e}")
    
    # Results dictionary
    results = {
        'accuracy': accuracy,
        'classification_report': clf_report,
        'confusion_matrix': cm,
        'roc_auc': roc_auc,
        'feature_importances': top_features,
        'figures': figures
    }
    
    return results

def analyze_outcome(data, outcome_column, outcome_columns, output_dir=None, cv_folds=5):
    """Analyze a single outcome column with optimized processing."""
    logger.info(f"\n{'=' * 80}\nANALYZING OUTCOME: {outcome_column}\n{'=' * 80}")
    
    # Prepare data
    X = data.drop(outcome_columns, axis=1)
    y = data[outcome_column]
    
    # Log class distribution
    class_dist = y.value_counts()
    class_balance = y.value_counts(normalize=True).round(2)
    logger.info(f"Class distribution for '{outcome_column}':\n{class_dist}")
    logger.info(f"Class balance: {class_balance}")
    
    # Preprocess features
    X = preprocess_features(X)
    feature_names = X.columns.tolist()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=feature_names
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=feature_names
    )
    
    # Define parameter grid - reduced for efficiency
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    # Grid search with optimizations
    logger.info("Performing Grid Search...")
    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
        param_grid=param_grid,
        cv=cv_folds,
        n_jobs=-1,
        scoring='f1',
        verbose=0,
        return_train_score=False  # Saves memory
    )
    
    # Fit grid search
    grid_search.fit(X_train_scaled, y_train)
    
    # Get best model
    best_params = grid_search.best_params_
    logger.info(f"Best parameters: {best_params}")
    
    # Train model with best parameters
    best_rf = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
    best_rf.fit(X_train_scaled, y_train)
    
    # Evaluate model
    results = evaluate_model(best_rf, X_test_scaled, y_test, outcome_column, feature_names)
    
    # Add best parameters to results
    results['best_params'] = best_params
    
    # Save model if output directory is provided
    if output_dir:
        import os
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, f"{outcome_column.replace(' ', '_')}_model.joblib")
        dump(best_rf, model_path)
        logger.info(f"Model saved to {model_path}")
    
    # Log results
    logger.info(f"Test accuracy: {results['accuracy']:.4f}")
    if results['roc_auc']:
        logger.info(f"ROC-AUC: {results['roc_auc']:.4f}")
    
    return best_rf, results

def main(data_path, outcome_columns, output_dir=None, parallel=True):
    """Main function to run the analysis pipeline."""
    # Load data
    data = load_and_prepare_data(data_path)
    
    # Analyze outcomes (parallel or sequential)
    results_dict = {}
    
    if parallel:
        # Parallel processing
        logger.info("Running analyses in parallel...")
        processed_results = Parallel(n_jobs=-1, verbose=10)(
            delayed(analyze_outcome)(data, outcome, outcome_columns, output_dir)
            for outcome in outcome_columns
        )
        
        # Store results
        for i, outcome in enumerate(outcome_columns):
            results_dict[outcome] = processed_results[i]
    else:
        # Sequential processing
        for outcome in outcome_columns:
            try:
                model, results = analyze_outcome(data, outcome, outcome_columns, output_dir)
                results_dict[outcome] = (model, results)
                logger.info(f"Analysis complete for '{outcome}'")
            except Exception as e:
                logger.error(f"Error analyzing '{outcome}': {e}")
                logger.info("Skipping this outcome and continuing with next one...")
    
    logger.info("All analyses complete!")
    return results_dict

if __name__ == "__main__":
    # Define outcome columns
    outcome_columns = [
        "Survival 6 months post crit care",
        "ECOG PS: 1=<2; 0=>3",
        "Oncology treatment, 0=no, 1=yes"
    ]
    
    # Run analysis
    results = main(
        data_path='data/Credit_ML_dataset_cleaned.xlsx',
        outcome_columns=outcome_columns,
        output_dir='results',
        parallel=True  # Set to False if running into memory issues
    )
    
    # Show plots - this will display all figures at once
    plt.show()
