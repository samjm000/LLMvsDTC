import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, f1_score
from sklearn.preprocessing import StandardScaler
import shap

# Set random seed for reproducibility
np.random.seed(42)

# Read the data from Excel file
data = pd.read_excel('data/Credit_ML_dataset_cleaned.xlsx')

# Check the data
print("Dataset shape:", data.shape)
print("\nColumns:", data.columns.tolist())
print("\nFirst few rows:")
print(data.head())

# Check for missing values
print("\nMissing values:")
print(data.isnull().sum())

# Define outcome columns
outcome_columns = [
    "Survival 6 months post crit care",
    "ECOG PS: 1=<2; 0=>3",
    "Oncology treatment, 0=no, 1=yes"
]

# Function to analyze each outcome
def analyze_outcome(data, outcome_column):
    print(f"\n\n{'=' * 80}")
    print(f"ANALYZING OUTCOME: {outcome_column}")
    print(f"{'=' * 80}")
    
    # Prepare the data
    # Separate features from target
    X = data.drop(outcome_columns, axis=1)
    y = data[outcome_column]
    
    # Check for categorical columns and handle them
    categorical_cols = X.select_dtypes(include=['object', 'bool']).columns
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    # Remove any remaining non-numeric columns
    X = X.select_dtypes(include=['int64', 'float64'])
    
    # Store feature names for later use
    feature_names = X.columns.tolist()
    
    # Print class distribution
    print(f"\nClass distribution for '{outcome_column}':")
    print(y.value_counts())
    print(f"Class balance: {y.value_counts(normalize=True).round(2)}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame to maintain column names
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=feature_names)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=feature_names)
    
    # Define the parameter grid for grid search
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # Create a base model
    rf = RandomForestClassifier(random_state=42)
    
    # Instantiate the grid search model
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        scoring='f1',
        verbose=1
    )
    
    # Fit the grid search to the data
    print("\nPerforming Grid Search...")
    grid_search.fit(X_train_scaled, y_train)
    
    # Get the best parameters
    best_params = grid_search.best_params_
    print(f"\nBest parameters: {best_params}")
    
    # Train model with best parameters
    best_rf = RandomForestClassifier(**best_params, random_state=42)
    best_rf.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = best_rf.predict(X_test_scaled)
    y_prob = best_rf.predict_proba(X_test_scaled)[:,1]
    
    # Evaluate the model
    print("\nModel Evaluation:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Calculate AUC-ROC
    try:
        roc_auc = roc_auc_score(y_test, y_prob)
        print(f"ROC-AUC Score: {roc_auc:.4f}")
    except Exception as e:
        print(f"ROC-AUC could not be calculated: {e}")
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f'Confusion Matrix - {outcome_column}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()
    
    # Plot ROC curve
    try:
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {outcome_column}')
        plt.legend()
        plt.show()
    except Exception as e:
        print(f"ROC curve could not be plotted: {e}")
    
    # Feature importance plot
    plt.figure(figsize=(12, 8))
    feature_importances = pd.DataFrame({
        'Feature': feature_names,
        'Importance': best_rf.feature_importances_
    })
    feature_importances = feature_importances.sort_values('Importance', ascending=False).head(20)
    sns.barplot(x='Importance', y='Feature', data=feature_importances)
    plt.title(f'Top 20 Feature Importances - {outcome_column}')
    plt.tight_layout()
    plt.show()
    
    print("\nGenerating SHAP values for model interpretation...")
    try:
        explainer = shap.TreeExplainer(best_rf)

        # Get a subset of test data for SHAP analysis
        shap_sample_size = min(20, X_test_scaled_df.shape[0])
        X_test_sample = X_test_scaled_df.iloc[:shap_sample_size]

        # Calculate SHAP values
        shap_values = explainer.shap_values(X_test_sample)

        # Debugging: Print SHAP output structure
        print("\nSHAP Values Type:", type(shap_values))
        print("SHAP Values Shape:", np.shape(shap_values))  # Check shape
        print("Expected Value Type:", type(explainer.expected_value))
        print("Expected Value:", explainer.expected_value)

        # Ensure binary classification handling
        if isinstance(shap_values, list):
            if len(shap_values) == 2:  # Binary classification
                shap_values_selected = shap_values[1]  # Use positive class SHAP values
                base_value = explainer.expected_value  # May be a list
            else:
                raise ValueError("Unexpected SHAP output structure for binary classification.")
        else:
            shap_values_selected = shap_values
            base_value = explainer.expected_value

        # 🚨 **Fix: Ensure base_value is a single scalar**
        if isinstance(base_value, (list, np.ndarray)):  
            base_value = base_value[1] if len(base_value) > 1 else base_value[0]  # Extract a single value

        # 🚨 **Fix: Ensure SHAP values are 1D**
        instance_idx = 0  # First test sample
        shap_values_instance = shap_values_selected[instance_idx]

        # Print debug info
        print("SHAP Instance Shape (Should be 1D):", shap_values_instance.shape)

        # Ensure it's **1D** for the waterfall plot
        if len(shap_values_instance.shape) > 1:
            shap_values_instance = shap_values_instance[:, 1]  # Extract only class 1

        # Create SHAP Explanation object
        shap_exp = shap.Explanation(
            values=shap_values_instance,
            base_values=base_value,
            data=X_test_sample.iloc[instance_idx].values,  # Ensure correct shape
            feature_names=X_test_sample.columns.tolist()
        )

        print(f"\nSHAP Base Value: {base_value}")
        print(f"Feature Contributions:\n{shap_exp}")

        # 🚀 **Plot waterfall correctly**
        plt.figure(figsize=(12, 8))
        shap.waterfall_plot(shap_exp)
        plt.title(f'SHAP Waterfall Plot (First Test Sample) - {outcome_column}')
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"SHAP analysis encountered an error: {e}")
        print("Skipping SHAP visualization and continuing with analysis...")





    
    # Return the model and results
    results = {
        'best_params': best_params,
        'accuracy': accuracy_score(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'confusion_matrix': cm,
        'feature_importances': feature_importances
    }
    
    try:
        results['roc_auc'] = roc_auc
    except:
        results['roc_auc'] = None
    
    return best_rf, results

# Analyze each outcome
for outcome in outcome_columns:
    try:
        model, results = analyze_outcome(data, outcome)
        print(f"\nAnalysis complete for '{outcome}'")
        print(f"Best parameters: {results['best_params']}")
        print(f"Test accuracy: {results['accuracy']:.4f}")
        if results.get('roc_auc'):
            print(f"ROC-AUC: {results['roc_auc']:.4f}")
    except Exception as e:
        print(f"Error analyzing '{outcome}': {e}")
        print("Skipping this outcome and continuing with next one...")
    
print("\nAll analyses complete!")
