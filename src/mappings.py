# utils/mappings.py

# Task 1: Complete Input Mapping Dictionary (Human-Readable to Model-Code)
# Corrected based on user feedback.

GERMAN_CREDIT_MAPPINGS = {
    # ----------------------------------------------------
    # Feature: Account_status (A1) - Removed "A14"
    # ----------------------------------------------------
    "Account_status": { # A1
        "< 0 DM": "A11",
        "0 <= ... < 200 DM": "A12",
        ">= 200 DM / salary assignments": "A13",
        # "No checking account": "A14" - REMOVED
    },
    
    # ----------------------------------------------------
    # Feature: Credit_history (A3)
    # ----------------------------------------------------
    "Credit_history": { # A3
        "No credits taken / all paid back duly": "A30",
        "All credits at this bank paid back duly": "A31",
        "Existing credits paid back duly till now": "A32",
        "Delay in paying off in the past": "A33",
        "Critical account / other credits existing": "A34"
    },

    # ----------------------------------------------------
    # Feature: loan_Purpose (A4)
    # ----------------------------------------------------
    "loan_Purpose": { # A4
        "Car (new)": "A40",
        "Car (used)": "A41",
        "Furniture / Equipment": "A42",
        "Radio / Television": "A43",
        "Domestic Appliances": "A44",
        "Repairs": "A45",
        "Education": "A46",
        "Vacation": "A47",
        "Business": "A49",
        "Others": "A410"
    },

    # ----------------------------------------------------
    # Feature: Savings/Bonds_AC (A6)
    # ----------------------------------------------------
    "Savings/Bonds_AC": { # A6
        "< 100 DM": "A61",
        "100 <= ... < 500 DM": "A62",
        "500 <= ... < 1000 DM": "A63",
        ">= 1000 DM": "A64",
        "Unknown / No savings account": "A65"
    },

    # ----------------------------------------------------
    # Feature: Present_Employment_since (A7)
    # ----------------------------------------------------
    "Present_Employment_since": { # A7
        "unemployed": "A71",
        "< 1 year": "A72",
        "1 <= ... < 4 years": "A73",
        "4 <= ... < 7 years": "A74",
        ">= 7 years": "A75"
    },
    
    # ----------------------------------------------------
    # Feature: Sex (A9) - Corrected to be separate
    # ----------------------------------------------------
    "Sex": { # Part of A9 (simplified)
        "Male": "Male",   
        "Female": "Female", 
    },
    
    # ----------------------------------------------------
    # Feature: Marital_status (A9) - Corrected, Removed "A95"
    # NOTE: The model must expect a combined code or your data prep handles this.
    # Standard A9 codes are for combined Sex/Marital Status. Using A9 codes here
    # assumes your preprocessor knows how to split them.
    # ----------------------------------------------------
    "Marital_status": { # Part of A9
        "Divorced/Separated": "A91",
        "Divorced/Separated/Married": "A92",
        "Single": "A93",
        "Married/Widowed": "A94",
        # "Female Single": "A95" - REMOVED
    },
    
    # ----------------------------------------------------
    # Feature: co-debtors (A10)
    # ----------------------------------------------------
    "co-debtors": { # A10
        "None": "A101",
        "Co-applicant": "A102",
        "Guarantor": "A103"
    },

    # ----------------------------------------------------
    # Feature: Assets/Physical_property (A12)
    # ----------------------------------------------------
    "Assets/Physical_property": { # A12
        "Real Estate": "A121",
        "Life Insurance / Building Society Savings": "A122",
        "Car or Other, Not in above": "A123",
        "No known assets / None": "A124"
    },
    
    # ----------------------------------------------------
    # Feature: Other_loans (A14)
    # ----------------------------------------------------
    "Other_loans": { # A14
        "Bank": "A141",
        "Stores": "A142",
        "None": "A143"
    },
    
    # ----------------------------------------------------
    # Feature: Housing (A15)
    # ----------------------------------------------------
    "Housing": { # A15
        "Rent": "A151",
        "Own": "A152",
        "Free": "A153"
    },
    
    # ----------------------------------------------------
    # Feature: Job_status (A17)
    # ----------------------------------------------------
    "Job_status": { # A17
        "Unemployed / Unskilled - Non-resident": "A171",
        "Unskilled - Resident": "A172",
        "Skilled Employee / Official": "A173",
        "Highly Skilled / Management": "A174"
    },
    
    # ----------------------------------------------------
    # Feature: Telephone (A19)
    # ----------------------------------------------------
    "Telephone": { # A19
        "None": "A191",
        "Yes (Registered)": "A192"
    },
    
    # ----------------------------------------------------
    # Feature: Foreign_worker (A20)
    # ----------------------------------------------------
    "Foreign_worker": { # A20
        "Yes": "A201",
        "No": "A202"
    }
}


# Task 2: Numerical Ranges and Defaults
NUMERICAL_RANGES = {
    "Duration_months": {"min": 4, "max": 72, "default": 18},
    "Credit_amount": {"min": 250, "max": 18424, "default": 4000},
    "Age": {"min": 19, "max": 75, "default": 35},
    "loan_wage_ratio": {"min": 1, "max": 4, "default": 3},
    "Tenure": {"min": 1, "max": 4, "default": 2}, 
    "Existing Credit": {"min": 1, "max": 4, "default": 1},
    "Dependents": {"min": 1, "max": 2, "default": 1}
}

# Mapping for the model's output (0/1) to human-readable labels
PREDICTION_MAPPING = {
    0: "✅ Low Credit Risk (Good Customer)",
    1: "❌ High Credit Risk (Bad Customer)"
}

# List of all 21 feature names in the EXACT order your model expects (CRITICAL)
MODEL_FEATURE_ORDER = [
    'Account_status', 'Duration_months', 'Credit_history', 'loan_Purpose',
    'Credit_amount', 'Savings/Bonds_AC', 'Present_Employment_since',
    'loan_wage_ratio', 'Sex', 'Marital_status', 'co-debtors', 'Tenure',
    'Assets/Physical_property', 'Age', 'Other_loans', 'Housing',
    'Existing Credit', 'Job_status', 'Dependents', 'Telephone',
    'Foreign_worker'
]