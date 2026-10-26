"""Dataset schema: column names, code -> label maps, and column groups."""

COLUMNS = [
    "checking_status", "duration_months", "credit_history", "purpose",
    "credit_amount", "savings", "employment_since", "installment_rate",
    "personal_status_sex", "other_debtors", "residence_since", "property",
    "age", "other_installment_plans", "housing", "existing_credits", "job",
    "dependents", "telephone", "foreign_worker",
]
RAW_COLUMNS = COLUMNS + ["target"]  # target: 1 = good, 2 = bad

CODE_MAPS = {
    "checking_status": {"A11": "< 0 DM", "A12": "0-200 DM", "A13": ">= 200 DM / salary assigned at least 1 yr",
                        "A14": "no checking account"},
    "credit_history": {"A30": "no credits / all paid", "A31": "all paid at this bank",
                       "A32": "existing credits paid duly", "A33": "past payment delay",
                       "A34": "critical / credits elsewhere"},
    "purpose": {"A40": "new car", "A41": "used car", "A42": "furniture/equipment", "A43": "radio/TV",
                "A44": "domestic appliances", "A45": "repairs", "A46": "education", "A47": "vacation",
                "A48": "retraining", "A49": "business", "A410": "others"},
    "savings": {"A61": "< 100 DM", "A62": "100-500 DM", "A63": "500-1000 DM", "A64": ">= 1000 DM",
                "A65": "unknown / none"},
    "employment_since": {"A71": "unemployed", "A72": "< 1 yr", "A73": "1-4 yrs", "A74": "4-7 yrs",
                         "A75": ">= 7 yrs"},
    "personal_status_sex": {"A91": "male divorced/separated", "A92": "female divorced/separated/married",
                            "A93": "male single", "A94": "male married/widowed", "A95": "female single"},
    "other_debtors": {"A101": "none", "A102": "co-applicant", "A103": "guarantor"},
    "property": {"A121": "real estate", "A122": "savings agreement / life insurance",
                 "A123": "car or other", "A124": "unknown / no property"},
    "other_installment_plans": {"A141": "bank", "A142": "stores", "A143": "none"},
    "housing": {"A151": "rent", "A152": "own", "A153": "for free"},
    "job": {"A171": "unemployed / unskilled non-resident", "A172": "unskilled resident",
            "A173": "skilled employee", "A174": "management / self-employed / highly qualified"},
    "telephone": {"A191": "none", "A192": "yes"},
    "foreign_worker": {"A201": "yes", "A202": "no"},
}
SEX_FROM_STATUS = {"A91": "male", "A92": "female", "A93": "male", "A94": "male", "A95": "female"}

CATEGORICAL_COLS = list(CODE_MAPS.keys())
NUMERIC_COLS = ["duration_months", "credit_amount", "installment_rate", "residence_since",
                "age", "existing_credits", "dependents"]
ENGINEERED_COLS = ["monthly_payment", "log_credit_amount"]
ALL_FEATURE_COLS = CATEGORICAL_COLS + NUMERIC_COLS + ENGINEERED_COLS
AUDIT_COLS = ["sex", "age_group"]   # derived, used for fairness audit (not features by default)
TARGET = "default"                   # 1 = bad, 0 = good (created from raw `target`)

AGE_BINS = [0, 25, 35, 50, 200]
AGE_LABELS = ["<25", "25-34", "35-49", "50+"]

# Orderings used when clustering (labels must match CODE_MAPS values)
ORDER_EMPLOYMENT = {"unemployed": 0, "< 1 yr": 1, "1-4 yrs": 2, "4-7 yrs": 3, ">= 7 yrs": 4}
ORDER_JOB = {"unemployed / unskilled non-resident": 0, "unskilled resident": 1,
             "skilled employee": 2, "management / self-employed / highly qualified": 3}
ORDER_SAVINGS = {"< 100 DM":0, "100-500 DM": 1, "500-1000 DM": 2, ">= 1000 DM": 3}
ORDER_CHECKING = {"< 0 DM": 0, "0-200 DM": 1, ">= 200 DM / salary assigned at least 1 yr": 2}

FRIENDLY = {
    "checking_status": "Checking account balance", "duration_months": "Loan duration",
    "credit_history": "Past repayment behaviour", "purpose": "Loan purpose",
    "credit_amount": "Loan amount", "savings": "Savings balance",
    "employment_since": "Time in current job", "installment_rate": "Instalment as % of income",
    "personal_status_sex": "Marital status / sex", "other_debtors": "Co-applicant / guarantor",
    "residence_since": "Years at residence", "property": "Most valuable asset",
    "age": "Age", "other_installment_plans": "Other credit obligations", "housing": "Housing",
    "existing_credits": "Credits at this bank", "job": "Job category",
    "dependents": "People to maintain", "telephone": "Registered phone",
    "foreign_worker": "Foreign worker", "monthly_payment": "Implied monthly payment",
    "log_credit_amount": "Loan amount (log)",
}
