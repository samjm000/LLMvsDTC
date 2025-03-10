LLM vs DTC: Machine Learning Edition
This repository contains the code and data for the "Doctors That Code (DTC) vs. Language Learning Models (LLM)" challenge, where I test whether modern LLMs can successfully create a machine learning pipeline without human intervention.
Project Overview
In this experiment, I used ChatGPT, Claude, and other LLMs to attempt to build a complete machine learning pipeline for analyzing critical care patient data. The goal was to determine if LLMs could:

Generate synthetic medical data
Preprocess and clean data with missing values
Build and optimize a Random Forest model
Create appropriate visualizations for model interpretation

The key questions we aimed to answer with our machine learning model were:

Will patients be fit for cancer treatment again?
Are patients in a good state of fitness when/if they are discharged?
Do patients survive for 6 months after being discharged?

Repository Structure

synthetic_data/: Contains the synthetic dataset generated for this experiment
preprocessing/: Code for data cleaning and preprocessing
models/: Random Forest implementation and optimization
visualizations/: Output plots and charts from the model
prompts/: The prompts used to generate code from various LLMs

Key Findings
This experiment demonstrated that:

LLMs can generate complex code for machine learning pipelines, but often require human intervention to fix errors and ensure logical consistency.
The code quality from LLMs varies significantly - sometimes producing clean, modular code, and other times creating monolithic, difficult-to-maintain solutions.
LLMs excel at suggesting advanced techniques (like KNN imputation) that might not be immediately obvious to the human programmer.
Human domain knowledge remains essential, particularly for medical data where relationships between variables are complex.

Running the Code
Requirements:

Python 3.8+
pandas
numpy
scikit-learn
matplotlib
seaborn
shap

To run the final optimized model:
Copypython models/random_forest_optimized.py
Blog Post
This project is detailed in my blog post: "Doctors That Code vs. LLMs: Machine Learning Edition" w
