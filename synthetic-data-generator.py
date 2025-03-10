import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_synthetic_data(n_samples=400):
    # Initialize empty dataframe
    data = {}
    
    # Age on admission: typically for cancer patients, range from 18-95
    data['Age on admission'] = np.random.randint(18, 96, n_samples)
    
    # Sex: M or F
    data['Sex'] = np.random.choice(['M', 'F'], n_samples, p=[0.5, 0.5])
    
    # BMI: normal distribution around mean of 26 with sd of 5
    data['BMI'] = np.round(np.random.normal(26, 5, n_samples), 2)
    # Ensure BMI is within reasonable limits (15-60)
    data['BMI'] = np.clip(data['BMI'], 15, 60)
    
    # ECOG PS at referral: 0-4
    data['ECOG PS at referral to Oncology'] = np.random.choice(range(0, 5), n_samples, 
                                                           p=[0.2, 0.3, 0.25, 0.15, 0.1])
    
    # ECOG PS on admission: 0-4, likely correlated with referral, but potentially worse
    ecog_ref = data['ECOG PS at referral to Oncology']
    data['ECOG PS on admission to hosptial'] = [min(4, max(0, x + np.random.choice([-1, 0, 1, 2], p=[0.05, 0.4, 0.4, 0.15]))) 
                                            for x in ecog_ref]
    


    n_samples = 400
    # Charlson Comorbidity Index: 0-16
    charlson_probs = [0.05, 0.10, 0.15, 0.15, 0.15, 0.10, 0.10, 0.05, 0.05, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01, 0.005, 0.005]
    # Make sure probabilities sum to 1
    charlson_probs[-1] += 1.0 - sum(charlson_probs)

    print(f"Probabilities sum to: {sum(charlson_probs)}")
    print(f"Probabilities: {charlson_probs}")


    # Create DataFrame
    data = pd.DataFrame()

    # Generate synthetic data
    data['Charleson Comorbidity Index'] = np.random.choice(range(0, 17), n_samples, p=charlson_probs)

    # Diagnosis categories
    diagnosis_categories = [
        'Urology', 'Lower GI', 'H&N', 'Breast', 'Lung', 'Brain', 'CUP', 
        'Upper GI', 'Gynae', 'Sarcoma', 'Endocrine', 'Melanoma', 
        'Breast and Lower GI', 'Germ Cell', 'Melanoma and Urology', 
        'Lung and breast', 'Squamous cell carcinoma', 'Unknown'
    ]
    
    diagnosis_probs = [0.15, 0.12, 0.08, 0.12, 0.1, 0.08, 0.04, 
                      0.08, 0.08, 0.04, 0.03, 0.04, 0.01, 0.01, 
                      0.005, 0.005, 0.005, 0.005]
    # Ensure probabilities sum to 1
    diagnosis_probs = [p/sum(diagnosis_probs) for p in diagnosis_probs]
    
    data['Diagnosis categories'] = np.random.choice(diagnosis_categories, n_samples, p=diagnosis_probs)
    
    # Most recent oncological treatment
    treatment_categories = [
        'Oral targeted therapy', 'Non-oral cytotoxic chemotherapy', 
        'Non-oral targeted therapy', 'Immunotherapy', 'Radioisotopes',
        'Oral cytotoxic chemotherapy', 'Radiotherapy', 'Other',
        'Chemoradiotherapy', 'Surgery'
    ]
    
    treatment_probs = [0.15, 0.25, 0.1, 0.1, 0.03, 0.07, 0.12, 0.05, 0.03, 0.1]
    # Ensure probabilities sum to 1
    treatment_probs = [p/sum(treatment_probs) for p in treatment_probs]
    
    data['Most recent oncological treatment'] = np.random.choice(treatment_categories, n_samples, p=treatment_probs)
    
    # Anticancer Therapy with 6 weeks: 0-1
    data['Anticancer Therapy with 6 weeks'] = np.random.choice([0, 1], n_samples, p=[0.3, 0.7])
    
    # Reason for admission to hospital
    admission_reasons = ['Disease related', 'Treatment related', 'Other']
    data['Reason for admission to hospital'] = np.random.choice(admission_reasons, n_samples, p=[0.6, 0.3, 0.1])
    
    # Surgical or medical
    data['Surgical or medical'] = np.random.choice(['Surgical', 'Medical'], n_samples, p=[0.25, 0.75])
    
    # Final NEWS 2 score Before Critical Care admission: 0-20, higher scores less common
    news_probs = np.array([0.05, 0.1, 0.15, 0.15, 0.15, 0.1, 0.08, 0.05, 0.05, 0.03, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01, 0.005, 0.005, 0.003, 0.002, 0.002])
    news_probs = news_probs / news_probs.sum()  # Normalize to ensure sum is 1.0
    data['Final NEWS 2 score Before Critical Care admission'] = np.random.choice(range(0, 21), n_samples, p=news_probs)
    
    # Temperature variables (in Celsius)
    data['Highest Temp in preceding 8 hours'] = np.round(np.random.normal(37.5, 1.2, n_samples), 1)
    data['Lowest Temp in preceding 8 hours'] = [max(35, min(x-np.random.uniform(0.1, 1.5), x)) 
                                           for x in data['Highest Temp in preceding 8 hours']]
    data['Lowest Temp in preceding 8 hours'] = np.round(data['Lowest Temp in preceding 8 hours'], 1)
    
    # Mean Arterial Pressure (MAP): typically 60-120
    data['MAP'] = np.round(np.random.normal(85, 15, n_samples))
    data['MAP'] = np.clip(data['MAP'], 55, 130)
    
    # Heart rate: typically 40-160
    data['Final HR before Critical Care admission'] = np.round(np.random.normal(90, 20, n_samples))
    data['Final HR before Critical Care admission'] = np.clip(data['Final HR before Critical Care admission'], 40, 180)
    
    # Cardiac arrest_1
    data['Cardiac arrest_1'] = np.random.choice(['No', 'Yes'], n_samples, p=[0.9, 0.1])
    
    # Respiratory rate: typically 8-40
    data['Final RR before Critical Care admission'] = np.round(np.random.normal(20, 6, n_samples))
    data['Final RR before Critical Care admission'] = np.clip(data['Final RR before Critical Care admission'], 6, 50)
    
    # Direct admission from theatre
    data['Direct admission from theatre? '] = np.random.choice(['No', 'Yes'], n_samples, p=[0.8, 0.2])
    
    # Features of sepsis
    data['Features of sepsis? '] = np.random.choice(['No', 'Yes', ''], n_samples, p=[0.5, 0.3, 0.2])
    
    gcs_probs = [0.05, 0.05, 0.05, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.05, 0.05, 0.05]

    # Make sure probabilities sum to 1.0
    gcs_probs[-1] += 1.0 - sum(gcs_probs)

    n_samples = 400

    data['First GCS on Critical Care admission'] = np.random.choice(
        range(3, 16),
        size=n_samples,
        p=gcs_probs
    )

    
    # Lowest temperature
    data['Lowest temp'] = np.round(np.random.normal(36, 1, n_samples), 2)
    data['Lowest temp'] = np.clip(data['Lowest temp'], 34, 37)
    
    # Highest and lowest heart rate
    data['Highest HR'] = np.round(np.random.normal(110, 20, n_samples))
    data['Highest HR'] = np.clip(data['Highest HR'], 60, 200)
    
    data['Lowest HR'] = [max(40, min(x-np.random.randint(10, 50), x)) for x in data['Highest HR']]
    data['Lowest HR'] = np.round(data['Lowest HR'])
    
    # Cardiac arrest_2
    data['Cardiac arrest_2'] = np.random.choice(['No', 'Yes'], n_samples, p=[0.85, 0.15])
    
    # Highest and lowest respiratory rate
    data['Highest RR'] = np.round(np.random.normal(28, 8, n_samples))
    data['Highest RR'] = np.clip(data['Highest RR'], 12, 60)
    
    data['Lowest RR'] = [max(6, min(x-np.random.randint(4, 20), x)) for x in data['Highest RR']]
    data['Lowest RR'] = np.round(data['Lowest RR'])
    
    # Lowest GCS
    data['Lowest GCS'] = [max(3, min(x-np.random.randint(0, 10), x)) for x in data['First GCS on Critical Care admission']]
    
    # Urine output
    data['Urine output ml per day'] = np.round(np.random.normal(1500, 800, n_samples))
    data['Urine output ml per day'] = np.clip(data['Urine output ml per day'], 0, 4000)
    # Format with commas for thousands
    data['Urine output ml per day'] = [f"{int(x):,}" for x in data['Urine output ml per day']]
    
    # Pressors y/n
    data['Pressors y/n'] = np.random.choice([0, 1], n_samples, p=[0.4, 0.6])
    
    # Mechanical ventilation
    data['Mechanical ventilation (incl CPAP)'] = np.random.choice(['No', 'Invasive', 'Non-invasive'], 
                                                              n_samples, p=[0.3, 0.4, 0.3])
    
    # Haemodialysis
    data['Haemodialysis /CRRT'] = np.random.choice(['No', 'Yes'], n_samples, p=[0.8, 0.2])
    
    # Hematology values
    data['Hb_1'] = np.round(np.random.normal(100, 20, n_samples))
    data['Hb_1'] = np.clip(data['Hb_1'], 50, 160)
    
    data['Haematocrit_1'] = np.round(data['Hb_1'] / 330, 2)
    
    data['WBC_1'] = np.round(np.random.normal(10, 5, n_samples), 2)
    data['WBC_1'] = np.clip(data['WBC_1'], 0.5, 30)
    
    data['Neutrophils_1'] = np.round(data['WBC_1'] * np.random.uniform(0.5, 0.9, n_samples), 2)
    
    data['Platelets_1'] = np.round(np.random.normal(250, 100, n_samples))
    data['Platelets_1'] = np.clip(data['Platelets_1'], 20, 600)
    
    # Biochemistry values
    data['Na_1'] = np.round(np.random.normal(138, 5, n_samples))
    data['Na_1'] = np.clip(data['Na_1'], 120, 155)
    
    data['K_1'] = np.round(np.random.normal(4.2, 0.6, n_samples), 1)
    data['K_1'] = np.clip(data['K_1'], 2.5, 7.0)
    
    data['Urea_1'] = np.round(np.random.normal(8, 5, n_samples), 1)
    data['Urea_1'] = np.clip(data['Urea_1'], 2, 40)
    
    data['Creatinine_umolperL_1'] = np.round(np.random.normal(100, 50, n_samples))
    data['Creatinine_umolperL_1'] = np.clip(data['Creatinine_umolperL_1'], 40, 500)
    
    # AKI y/n
    data['AKI y/n'] = np.random.choice([0, 1], n_samples, p=[0.6, 0.4])
    
    # Liver function
    data['Bilirubin_1'] = np.round(np.random.normal(15, 10, n_samples))
    data['Bilirubin_1'] = np.clip(data['Bilirubin_1'], 3, 100)
    
    data['Albumin_1'] = np.round(np.random.normal(30, 5, n_samples))
    data['Albumin_1'] = np.clip(data['Albumin_1'], 15, 45)
    
    # Second set of hematology values (slightly different from first set)
    data['Hb_2'] = data['Hb_1'] + np.random.normal(-5, 10, n_samples)
    data['Hb_2'] = np.round(np.clip(data['Hb_2'], 45, 160))
    
    data['Haematocrit_2'] = np.round(data['Hb_2'] / 330, 2)
    
    data['WBC_2'] = data['WBC_1'] + np.random.normal(0, 2, n_samples)
    data['WBC_2'] = np.round(np.clip(data['WBC_2'], 0.5, 30), 1)
    
    data['Platelets_2'] = data['Platelets_1'] + np.random.normal(-20, 40, n_samples)
    data['Platelets_2'] = np.round(np.clip(data['Platelets_2'], 10, 600))
    
    # Second set of biochemistry
    data['Na_2'] = data['Na_1'] + np.random.normal(0, 3, n_samples)
    data['Na_2'] = np.round(np.clip(data['Na_2'], 120, 155))
    
    data['K_2'] = data['K_1'] + np.random.normal(0, 0.5, n_samples)
    data['K_2'] = np.round(np.clip(data['K_2'], 2.5, 7.0), 1)
    
    data['Urea_2'] = data['Urea_1'] + np.random.normal(1, 2, n_samples)
    data['Urea_2'] = np.round(np.clip(data['Urea_2'], 2, 45), 1)
    
    # Creatinine in mg/dL (conversion from μmol/L)
    data['Creatinine_mgperdL_2'] = np.round(data['Creatinine_umolperL_1'] / 88.4, 2)
    
    data['Creatinine_umolperL_2'] = data['Creatinine_umolperL_1'] + np.random.normal(10, 20, n_samples)
    data['Creatinine_umolperL_2'] = np.round(np.clip(data['Creatinine_umolperL_2'], 40, 550))
    
    # Acute renal failure
    data['Acute renal failure_2'] = np.random.choice(['NO', 'YES'], n_samples, p=[0.7, 0.3])
    
    # More liver function
    data['Albumin_2'] = data['Albumin_1'] + np.random.normal(-2, 3, n_samples)
    data['Albumin_2'] = np.round(np.clip(data['Albumin_2'], 10, 45))
    
    data['Bilirubin_2'] = data['Bilirubin_1'] + np.random.normal(3, 8, n_samples)
    data['Bilirubin_2'] = np.round(np.clip(data['Bilirubin_2'], 3, 120))
    
    # Blood gas variables
    data['First pH on Admission to Critical Care'] = np.round(np.random.normal(7.35, 0.1, n_samples), 2)
    data['First pH on Admission to Critical Care'] = np.clip(data['First pH on Admission to Critical Care'], 7.0, 7.6)
    
    data['FiO2_1'] = np.random.choice([0.21, 0.28, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], n_samples)
    
    data['PaCo2 kPa_1'] = np.round(np.random.normal(5.0, 1.0, n_samples), 2)
    data['PaCo2 kPa_1'] = np.clip(data['PaCo2 kPa_1'], 2.5, 10.0)
    
    data['PaO2 kPa_1'] = np.round(np.random.normal(12.0, 5.0, n_samples), 2)
    data['PaO2 kPa_1'] = np.clip(data['PaO2 kPa_1'], 6.0, 30.0)
    
    # Calculate A-a gradient (simplified)
    data['Aa gradient_1'] = np.round(data['FiO2_1'] * 100 - data['PaO2 kPa_1'] * 5, 2)
    data['Aa gradient_1'] = np.clip(data['Aa gradient_1'], 0, 500)
    
    # Convert kPa to mmHg
    data['PaO2 mmHg_1'] = np.round(data['PaO2 kPa_1'] * 7.50062, 2)
    
    # Calculate P/F ratio
    data['PaO2_FiO2_1'] = np.round(data['PaO2 mmHg_1'] / data['FiO2_1'], 2)
    
    data['BiCarb_1'] = np.round(np.random.normal(24, 5, n_samples), 1)
    data['BiCarb_1'] = np.clip(data['BiCarb_1'], 10, 40)
    
    data['Lactate_1'] = np.round(np.random.normal(2, 1.5, n_samples), 1)
    data['Lactate_1'] = np.clip(data['Lactate_1'], 0.5, 15)
    
    # More FiO2 and blood gas values (follow-up measurements)
    data['FiO2_2'] = data['FiO2_1'] + np.random.choice([-0.1, 0, 0.1, 0.2], n_samples, p=[0.2, 0.4, 0.3, 0.1])
    data['FiO2_2'] = np.clip(data['FiO2_2'], 0.21, 1.0)
    data['FiO2_2'] = np.round(data['FiO2_2'] * 100) / 100
    
    data['pH_1'] = data['First pH on Admission to Critical Care']
    
    data['FiO2_3'] = data['FiO2_2']
    
    data['PaCo2 kPa_2'] = data['PaCo2 kPa_1'] + np.random.normal(0, 0.5, n_samples)
    data['PaCo2 kPa_2'] = np.round(np.clip(data['PaCo2 kPa_2'], 2.5, 10.0), 2)
    
    data['PaO2 kPa_2'] = data['PaO2 kPa_1'] + np.random.normal(0, 3.0, n_samples)
    data['PaO2 kPa_2'] = np.round(np.clip(data['PaO2 kPa_2'], 4.0, 40.0), 2)
    
    # Recalculate A-a gradient
    data['Aa gradient_2'] = np.round(data['FiO2_2'] * 100 - data['PaO2 kPa_2'] * 5, 2)
    data['Aa gradient_2'] = np.clip(data['Aa gradient_2'], 0, 500)
    
    # Convert kPa to mmHg again
    data['PaO2 mmHg_2'] = np.round(data['PaO2 kPa_2'] * 7.50062, 2)
    
    # Recalculate P/F ratio
    data['PaO2_FiO2_2'] = np.round(data['PaO2 mmHg_2'] / data['FiO2_2'], 2)
    
    # Worst P/F ratio (lower of the two)
    data['Worst PaO2:FiO2 ratio'] = np.minimum(data['PaO2_FiO2_1'], data['PaO2_FiO2_2'])
    data['Worst PaO2:FiO2 ratio'] = np.round(data['Worst PaO2:FiO2 ratio'], 2)
    
    data['BiCarb_2'] = data['BiCarb_1'] + np.random.normal(-1, 2, n_samples)
    data['BiCarb_2'] = np.round(np.clip(data['BiCarb_2'], 8, 40), 1)
    
    data['FiO2 _4'] = data['FiO2_2']
    
    p = [0.1, 0.1, 0.15, 0.15, 0.15, 
        0.1, 0.1, 0.05, 0.05, 0.02, 
        0.02, 0.01, 0.01, 0.01, 0.01, 
        0.005, 0.015]

    # Force sum to 1
    p[-1] += 1.0 - sum(p)

    # Safeguard to prevent negatives
    if p[-1] < 0:
        print(f"Warning: Last probability went negative. Clipping to 0.")
        p[-1] = 0.0

    # Optional normalization if anything was clipped
    total_p = sum(p)
    if total_p != 1.0:
        p = [x / total_p for x in p]

    print(f"Probabilities sum to: {sum(p)}")
    print(f"Minimum probability: {min(p)}")

    
    # Oncology treatment
    data['Oncology treatment, 0=no, 1=yes'] = np.random.choice([0, 1], n_samples, p=[0.3, 0.7])
    
    # Survival time (in days)
    survival_days = np.random.exponential(scale=180, size=n_samples)
    survival_days = np.round(survival_days).astype(int)
    survival_days = np.clip(survival_days, 1, 1000)
    data['Overall survival (including censor dates)'] = survival_days
    
    # 6-month survival
    data['Survival 6 months post crit care'] = ['Yes' if days > 180 else 'No' for days in survival_days]
    
    # Ensure 'ECOG PS on admission to hospital' is created
    data['ECOG PS on admission to hospital'] = np.random.choice(range(0, 5), n_samples, p=[0.2, 0.3, 0.3, 0.15, 0.05])

    # Use the correct column name (fix the typo)
    data['ECOG PS: 0=<2; 1=>3'] = [0 if ecog <= 2 else 1 for ecog in data['ECOG PS on admission to hospital']]


    # Create pandas DataFrame
    df = pd.DataFrame(data)
    
    return df

# Generate the data
synthetic_df = generate_synthetic_data(400)

# Save to CSV
synthetic_df.to_csv('synthetic_cancer_critical_care_data.csv', index=False)

print(f"Created synthetic dataset with {len(synthetic_df)} rows")