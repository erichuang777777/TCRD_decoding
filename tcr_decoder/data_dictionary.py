"""
Auto-generate a data dictionary for the decoded clinical output.

Produces a DataFrame describing every column: name, source, type,
unique values, missing rate, and clinical notes.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional


# Registry of output columns → source field + metadata
# Format: 'OutputColumn': ('source_raw_field', 'decoder_type', 'description')
COLUMN_REGISTRY: Dict[str, tuple] = {
    'Patient_ID': ('PK', 'passthrough', 'Unique patient identifier'),
    'Sex': ('SEX', 'mapping', 'Patient sex'),
    'Age_at_Diagnosis': ('AGE', 'numeric_clean', 'Age in years at cancer diagnosis'),
    'Diagnosis_Year': ('DX_YEAR', 'passthrough', 'Year of cancer diagnosis'),
    'Date_of_Diagnosis': ('DXDATE', 'date_clean', 'Date of cancer diagnosis (partial dates have day=99)'),
    'Date_of_First_Visit': ('VISTDATE', 'date_clean', 'Date of first visit to reporting hospital'),
    'Smoking': ('SMOKING', 'decode_smoking_triplet', 'Smoking/betelnut/alcohol: XX,XX,XX triplet decoded'),
    'Total_Primaries': ('SEQ1', 'mapping', 'Total number of primary cancers for this patient'),
    'Cancer_Sequence': ('SEQ2', 'mapping', 'Sequence number of this cancer (1st, 2nd, etc.)'),
    'Primary_Site_Code': ('TCODE1', 'passthrough', 'ICD-O-3 topography code (C50.x for breast)'),
    'Primary_Site': ('TCODE1', 'mapping', 'Primary site anatomical description'),
    'Laterality': ('LAT95', 'mapping', 'Left/Right/Bilateral'),
    'Histology_Code': ('MCODE', 'passthrough', 'ICD-O-3 morphology code'),
    'Histology': ('MCODE', 'mapping', 'Histological type description'),
    'Behavior': ('MCODE5', 'mapping', 'Behavior code (malignant, in-situ, etc.)'),
    'Grade_Pathologic': ('MCODE6', 'mapping', 'Pathologic grade/differentiation (BRS score)'),
    'Grade_Clinical': ('MCODE6C', 'mapping', 'Clinical grade/differentiation'),
    'Confirmation_Method': ('CONFER', 'mapping', 'Method of cancer diagnosis confirmation'),
    'Tumor_Size_mm': ('CSIZE95', 'numeric_clean', 'Tumor size in millimeters (999/888=unknown/NA)'),
    'Perineural_Invasion': ('PNI', 'mapping', 'Perineural invasion status'),
    'LVI': ('LVI', 'mapping', 'Lymphovascular invasion status'),
    'LN_Examined': ('LNEXAM', 'numeric_clean', 'Number of regional LN examined (95-99=sentinel codes)'),
    'LN_Positive': ('LN_POSITI', 'decode_lnpositive', 'LN positive count or status text'),
    'LN_Positive_Count': ('LN_POSITI', 'derived_numeric', 'Numeric LN positive count (NaN for text statuses)'),
    'AJCC_Edition': ('AJCC', 'mapping', 'AJCC Cancer Staging Edition used'),
    'Clinical_T': ('CT', 'mapping+fix', 'Clinical T stage'),
    'Clinical_N': ('CN', 'mapping+fix', 'Clinical N stage (N0 subcodes corrected)'),
    'Clinical_M': ('CM', 'mapping+fix', 'Clinical M stage (M0(i+) clarified)'),
    'Clinical_Stage': ('CSTG', 'mapping', 'Clinical AJCC stage group'),
    'Path_T': ('PT', 'mapping+fix', 'Pathologic T stage (post-neoadjuvant=NA)'),
    'Path_N': ('PN', 'mapping+fix', 'Pathologic N stage (0A=N0(i-), 0B=N0(i+))'),
    'Path_M': ('PM', 'mapping+fix', 'Pathologic M stage (B=M0(i+) ITC)'),
    'Path_Stage': ('PSTG', 'mapping+fix', 'Pathologic AJCC stage group'),
    'Combined_Stage': ('SUMSTG', 'mapping', 'Combined clinical+pathologic stage'),
    'Other_Staging_System': ('OSTG', 'mapping', 'Other staging system used (0=none)'),
    'Other_Clinical_Stage': ('OCSTG', 'mapping+crossfield', 'Other system clinical stage (0+OSTG=0→NA)'),
    'Other_Path_Stage': ('OPSTG', 'mapping+crossfield', 'Other system pathologic stage'),
    'Metastasis_Site_1': ('META1', 'mapping', 'First distant metastasis site'),
    'Metastasis_Site_2': ('META2', 'mapping', 'Second distant metastasis site'),
    'Metastasis_Site_3': ('META3', 'mapping', 'Third distant metastasis site'),
    'Surgery_Performed': ('S', 'mapping', 'Surgery at THIS hospital (Yes/No)'),
    'Surgery_Date': ('FSDATE', 'date_clean', 'Date of first surgery'),
    'Surgery_Type_Other_Hosp': ('PRESTYPE', 'mapping', 'Surgery type at OTHER hospital'),
    'Surgery_Type_This_Hosp': ('STYPE95', 'mapping', 'Surgery type at THIS hospital'),
    'Any_Surgery': ('S+PRESTYPE', 'derived_logic', 'Surgery at ANY hospital (Yes/No)'),
    'Minimally_Invasive': ('MINS', 'mapping', 'Minimally invasive surgery indicator'),
    'Surgical_Margin': ('MARG95', 'mapping', 'Surgical margin status'),
    'Surgical_Margin_mm': ('MARGDIS', 'numeric_clean', 'Surgical margin distance in mm'),
    'Regional_LN_Surgery_Other': ('PRESLNSCO', 'mapping', 'LN surgery at other hospital'),
    'Regional_LN_Surgery_This': ('SLNSCO95', 'mapping', 'LN surgery at this hospital'),
    'Sentinel_LN_Examined': ('SSF4', 'decode_sentinel', 'Sentinel LN examined count'),
    'Sentinel_LN_Positive': ('SSF5', 'decode_sentinel', 'Sentinel LN positive count'),
    'Radiation_Performed': ('R', 'mapping', 'Radiation therapy performed (this hospital)'),
    'RT_Target_Summary': ('RTAR', 'mapping', 'RT target volume summary (additive: 1=T, 2=N, 4=M)'),
    'RT_Modality': ('RMOD', 'mapping', 'Radiation modality (EBRT, brachytherapy, etc.)'),
    'EBRT_Technique': ('EBRT', 'decode_ebrt_additive', 'EBRT technique (additive coding: 1+2+4+8+32+64)'),
    'High_Dose_Target': ('HTAR', 'mapping', 'High-dose radiation target volume'),
    'High_Dose_cGy': ('HDOSE', 'numeric_clean', 'High-dose radiation dose in cGy'),
    'High_Dose_Fractions': ('HNO', 'numeric_clean', 'High-dose radiation fraction count'),
    'Low_Dose_Target': ('LTAR', 'mapping', 'Low-dose radiation target volume'),
    'Low_Dose_cGy': ('LDOSE', 'numeric_clean', 'Low-dose radiation dose in cGy'),
    'Low_Dose_Fractions': ('LNO', 'numeric_clean', 'Low-dose radiation fraction count'),
    'RT_Seq_Surgery': ('SEQRS', 'mapping', 'Sequence of RT relative to surgery'),
    'RT_vs_Systemic_Seq': ('SEQLS', 'mapping', 'Sequence of local vs systemic therapy'),
    'Chemo_Other_Hosp': ('PREC', 'mapping', 'Chemotherapy at other hospital'),
    'Chemo_This_Hosp': ('C', 'mapping', 'Chemotherapy at this hospital'),
    'Hormone_Other_Hosp': ('PREH', 'mapping', 'Hormone therapy at other hospital'),
    'Hormone_This_Hosp': ('H', 'mapping', 'Hormone therapy at this hospital'),
    'Immuno_Other_Hosp': ('PREI', 'mapping', 'Immunotherapy at other hospital'),
    'Immuno_This_Hosp': ('I', 'mapping', 'Immunotherapy at this hospital'),
    'Stem_Cell_Other_Hosp': ('PREB', 'mapping', 'BMT/SCT at other hospital'),
    'Stem_Cell_This_Hosp': ('B', 'mapping', 'BMT/SCT at this hospital'),
    'Targeted_Other_Hosp': ('PRETAR', 'mapping', 'Targeted therapy at other hospital'),
    'Targeted_This_Hosp': ('TAR', 'mapping', 'Targeted therapy at this hospital'),
    'Other_Treatment': ('OTH', 'mapping', 'Other treatment modality'),
    'Palliative_Care': ('PREP', 'mapping', 'Palliative care status'),
    'Active_Surveillance': ('WATCHWAITING', 'mapping', 'Active surveillance / watchful waiting'),
    'ER_Status': ('SSF1', 'decode_er_pr', 'Estrogen receptor status (code 888=converted Neg→Pos post-neoadjuvant)'),
    'PR_Status': ('SSF2', 'decode_er_pr', 'Progesterone receptor status'),
    'Neoadjuvant_Response': ('SSF3', 'decode_ssf3_neoadj', 'Neoadjuvant therapy response (010=cCR, 011=pCR)'),
    'Nottingham_Grade': ('SSF6', 'decode_nottingham', 'Nottingham/BR combined score and grade'),
    'HER2_Status': ('SSF7', 'decode_her2', 'HER2 IHC+ISH combined status (3-digit code)'),
    'Pagets_Disease': ('SSF8', 'mapping', "Paget's disease of nipple"),
    'LVI_SSF': ('SSF9', 'mapping', 'Lymphovascular invasion (SSF source)'),
    'Ki67_Index': ('SSF10', 'decode_ki67', 'Ki-67 proliferation index with Low/Intermediate/High'),
    'Vital_Status': ('VSTA', 'mapping', 'Vital status at initial follow-up'),
    'Cancer_Status': ('CSTA', 'mapping', 'Cancer status at initial follow-up'),
    'Last_Contact_Date': ('LCD', 'date_clean', 'Last contact date (initial follow-up)'),
    'Recurrence_Date': ('REDATE', 'date_clean', 'Recurrence date (initial follow-up)'),
    'Recurrence_Type': ('RETYPE95', 'mapping', 'Recurrence type (initial follow-up)'),
    'Cause_of_Death': ('DIECAUSE', 'decode_cod', 'Cause of death (initial follow-up)'),
    'Vital_Status_Extended': ('VSTA6', 'mapping', 'Vital status at extended follow-up'),
    'Last_Contact_Extended': ('LCD6', 'date_clean', 'Last contact date (extended follow-up)'),
    'Survival_Years': ('SURVY6+DXDATE+LCD6', 'derived_calc', 'Overall survival in years (recalculated from dates)'),
    'Recurrence_Date_Extended': ('REDATE6', 'date_clean', 'Recurrence date (extended follow-up)'),
    'Recurrence_Type_Extended': ('RETYPE6', 'mapping', 'Recurrence type (extended follow-up)'),
    'Cause_of_Death_Extended': ('DIECAUSE6', 'decode_cod', 'Cause of death (extended follow-up)'),
    'Height_cm': ('HEIGHT', 'numeric_clean', 'Height in cm (999=unknown)'),
    'Weight_kg': ('WEIGHT', 'numeric_clean', 'Weight in kg (999=unknown)'),
    'Performance_Status': ('KPSECOG', 'mapping', 'ECOG/KPS performance status'),
    'Class_of_Case': ('CLASS95', 'mapping', 'Class of case (relationship to reporting hospital)'),
    'Diag_at_Hosp': ('CLASSOFDIAG', 'mapping', 'Diagnosis location relative to reporting hospital'),
    'Treat_at_Hosp': ('CLASSOFTREAT', 'mapping', 'Treatment location relative to reporting hospital'),
    'Treatment_Data_Incomplete': ('CLASS95', 'derived_logic', 'True if Dx & Tx elsewhere (treatment data unreliable)'),
    # Derived variables (added by derived.py)
    'BMI': ('HEIGHT+WEIGHT', 'derived_calc', 'Body Mass Index (kg/m²)'),
    'BMI_Category': ('BMI', 'derived_cat', 'WHO BMI category'),
    'Age_Group': ('AGE', 'derived_cat', 'Age group (<40, 40-49, 50-59, 60-69, 70-79, ≥80)'),
    'Age_Group_Binary': ('AGE', 'derived_cat', 'Age ≤50 vs >50'),
    'OS_Months': ('Survival_Years', 'derived_calc', 'Overall survival in months'),
    'OS_Event': ('VSTA6', 'derived_logic', '1=dead (any cause), 0=alive/censored'),
    'CSS_Event': ('DIECAUSE6', 'derived_logic', '1=cancer-specific death, 0=other'),
    'RFS_Months': ('DXDATE+REDATE6+LCD6', 'derived_calc', 'Recurrence-free survival in months'),
    'RFS_Event': ('RETYPE6+VSTA6', 'derived_logic', '1=recurrence or death, 0=censored'),
    'Any_Chemotherapy': ('C+PREC', 'derived_logic', 'Chemotherapy at any hospital'),
    'Any_Radiation': ('R', 'derived_logic', 'Radiation at any hospital'),
    'Any_Hormone_Therapy': ('H+PREH', 'derived_logic', 'Hormone therapy at any hospital'),
    'Any_Targeted_Therapy': ('TAR+PRETAR', 'derived_logic', 'Targeted therapy at any hospital'),
    'Any_Immunotherapy': ('I+PREI', 'derived_logic', 'Immunotherapy at any hospital'),
    'Treatment_Modality_Count': ('multiple', 'derived_calc', 'Count of distinct treatment modalities (0-5)'),
    'Stage_Simple': ('PSTG', 'derived_cat', 'Simplified stage (I/II/III/IV)'),
    'T_Simple': ('PT', 'derived_extract', 'Simplified T stage (T0-T4)'),
    'N_Simple': ('PN', 'derived_extract', 'Simplified N stage (N0-N3)'),
    'M_Simple': ('PM', 'derived_extract', 'Simplified M stage (M0/M1)'),
    # LN_Ratio removed — not generated (ratio is clinically uninformative for registry use)
    'ER_Percent': ('SSF1', 'derived_extract', 'ER numeric percentage for continuous analysis'),
    'PR_Percent': ('SSF2', 'derived_extract', 'PR numeric percentage for continuous analysis'),
    'Dx_to_Surgery_Days': ('DXDATE+FSDATE', 'derived_calc', 'Days from diagnosis to surgery'),
    # ─── Breast-specific prognostic scores (derived.py) ────────────────────────
    # PEPI score: Ellis MJ et al. J Natl Cancer Inst. 2008;100:1380-1388
    'PEPI_pT_Points':        ('PT', 'derived_calc', 'PEPI score contribution from pT stage (0 or 3)'),
    'PEPI_pN_Points':        ('PN', 'derived_calc', 'PEPI score contribution from pN stage (0 or 3)'),
    'PEPI_Ki67_RFS_Points':  ('SSF10', 'derived_calc', 'PEPI Ki67 points for RFS (0-3)'),
    'PEPI_Ki67_BCSS_Points': ('SSF10', 'derived_calc', 'PEPI Ki67 points for BCSS (0-3; slightly higher than RFS)'),
    'PEPI_ER_Points':        ('SSF1', 'derived_calc', 'PEPI ER Allred proxy points (0 if ER%≥1, 3 if ER%<1)'),
    'PEPI_RFS_Score':        ('PT+PN+SSF10+SSF1', 'derived_calc',
                              'Total PEPI score for RFS (0=Best, 1-3=Intermediate, ≥4=Worse)'),
    'PEPI_BCSS_Score':       ('PT+PN+SSF10+SSF1', 'derived_calc',
                              'Total PEPI score for BCSS'),
    'PEPI_RFS_Group':        ('PEPI_RFS_Score', 'derived_cat',
                              'PEPI RFS risk group: Best/Intermediate/Worse'),
    'PEPI_BCSS_Group':       ('PEPI_BCSS_Score', 'derived_cat',
                              'PEPI BCSS risk group: Best/Intermediate/Worse'),
    # IHC4 score: Cuzick J et al. J Clin Oncol. 2011;29:4273-4278
    'IHC4_Score':            ('SSF1+SSF2+SSF10+SSF7', 'derived_calc',
                              'IHC4 score (94.7×formula); higher = greater 10-yr distant recurrence risk'),
    # NPI: Galea MH et al. Breast Cancer Res Treat. 1992;22:207-219
    'NPI_Score':             ('CSIZE95+LN_POSITI+SSF6', 'derived_calc',
                              'Nottingham Prognostic Index (0.2×size_cm + LN_stage + grade)'),
    'NPI_Group':             ('NPI_Score', 'derived_cat',
                              'NPI risk group (Blamey 2007 six-group): '
                              'Excellent/Good/Moderate I/Moderate II/Poor/Very Poor'),
    # Molecular subtype: St. Gallen 2013 consensus
    'Molecular_Subtype':     ('SSF1+SSF2+SSF7+SSF10', 'derived_logic',
                              'IHC surrogate molecular subtype (Luminal A/B, HER2-Enriched, TNBC)'),
}


# ─── TCR Official Field Sequence Numbers ─────────────────────────────────────
# Maps output column name → TCR codebook field number (欄位序號)
# Source: Longform-Manual 2025, Table 1 (pp. 12-16)
# Derived / calculated columns have no TCR field number and are omitted.
TCR_FIELD_NUMBER: Dict[str, str] = {
    # Section 1: Basic identifiers
    'Patient_ID':                 '1.1+1.2',
    'Sex':                        '1.5',
    # Section 2: Diagnosis info
    'Age_at_Diagnosis':           '2.1',
    'Total_Primaries':            '2.2',
    'Cancer_Sequence':            '2.2',
    'Class_of_Case':              '2.3',
    'Diag_at_Hosp':               '2.3.1',
    'Treat_at_Hosp':              '2.3.2',
    'Treatment_Data_Incomplete':  '2.3',
    'Date_of_First_Visit':        '2.4',
    'Date_of_Diagnosis':          '2.5',
    'Diagnosis_Year':             '2.5',
    'Primary_Site_Code':          '2.6',
    'Primary_Site':               '2.6',
    'Laterality':                 '2.7',
    'Histology_Code':             '2.8',
    'Histology':                  '2.8',
    'Behavior':                   '2.9',
    'Grade_Clinical':             '2.10.1',
    'Grade_Pathologic':           '2.10.2',
    'Confirmation_Method':        '2.11',
    'Tumor_Size_mm':              '2.13',
    'Perineural_Invasion':        '2.13.1',
    'LVI':                        '2.13.2',
    'LN_Examined':                '2.14',
    'LN_Positive':                '2.15',
    'LN_Positive_Count':          '2.15',
    # Section 3: Staging
    'Clinical_T':                 '3.4',
    'Clinical_N':                 '3.5',
    'Clinical_M':                 '3.6',
    'Clinical_Stage':             '3.7',
    'Path_T':                     '3.10',
    'Path_N':                     '3.11',
    'Path_M':                     '3.12',
    'Path_Stage':                 '3.13',
    'Combined_Stage':             '3.13',
    'AJCC_Edition':               '3.16',
    'Other_Staging_System':       '3.17',
    'Other_Clinical_Stage':       '3.19',
    'Other_Path_Stage':           '3.21',
    'Metastasis_Site_1':          '3.22',
    'Metastasis_Site_2':          '3.22',
    'Metastasis_Site_3':          '3.22',
    # Section 4: Treatment — Surgery
    'Surgery_Performed':          '4.1',
    'Surgery_Date':               '4.1.1',
    'Surgery_Type_Other_Hosp':    '4.1.3',
    'Surgery_Type_This_Hosp':     '4.1.4',
    'Any_Surgery':                '4.1',
    'Minimally_Invasive':         '4.1.4.1',
    'Surgical_Margin':            '4.1.5',
    'Surgical_Margin_mm':         '4.1.5.1',
    'Regional_LN_Surgery_Other':  '4.1.6',
    'Regional_LN_Surgery_This':   '4.1.7',
    # Section 4.2: Treatment — Radiation
    'Radiation_Performed':        '4.2',
    'RT_Target_Summary':          '4.2.1.1',
    'RT_Modality':                '4.2.1.2',
    'RT_Seq_Surgery':             '4.2.1.5',
    'RT_vs_Systemic_Seq':         '4.2.1.6',
    'EBRT_Technique':             '4.2.2.1',
    'High_Dose_cGy':              '4.2.2.4',
    'High_Dose_Fractions':        '4.2.2.5',
    'Low_Dose_cGy':               '4.2.2.7',
    'Low_Dose_Fractions':         '4.2.2.8',
    # Section 4.3: Systemic therapy
    'Chemo_This_Hosp':            '4.3',
    'Chemo_Other_Hosp':           '4.3',
    'Hormone_This_Hosp':          '4.4',
    'Hormone_Other_Hosp':         '4.4',
    'Targeted_This_Hosp':         '4.5',
    'Targeted_Other_Hosp':        '4.5',
    'Immuno_This_Hosp':           '4.6',
    'Immuno_Other_Hosp':          '4.6',
    # Section 5: Follow-up
    'Vital_Status':               '5.1',
    'Cancer_Status':              '5.3',
    'Last_Contact_Date':          '5.4',
    'Recurrence_Date':            '5.5',
    'Recurrence_Type':            '5.6',
    'Cause_of_Death':             '5.7',
    'Vital_Status_Extended':      '5.1',
    'Last_Contact_Extended':      '5.4',
    'Survival_Years':             '5.4+2.5',
    'Recurrence_Date_Extended':   '5.5',
    'Recurrence_Type_Extended':   '5.6',
    'Cause_of_Death_Extended':    '5.7',
    # Section 6: Physical measurements
    'Height_cm':                  '6.1',
    'Weight_kg':                  '6.2',
    'Smoking':                    '6.3',
    # Section 7: Performance status
    'Performance_Status':         '7.1',
    # Section 8: SSF fields (cancer-specific biomarkers)
    'ER_Status':                  '8.1',
    'PR_Status':                  '8.2',
    'Neoadjuvant_Response':       '8.3',
    'Sentinel_LN_Examined':       '8.4',
    'Sentinel_LN_Positive':       '8.5',
    'Nottingham_Grade':           '8.6',
    'HER2_Status':                '8.7',
    'Ki67_Index':                 '8.10',
    # SSF fields for other cancer types (8.x, cancer-specific)
    'EGFR_Mutation':              '8.6',
    'ALK_Translocation':          '8.7',
    'Separate_Tumor_Nodules':     '8.1',
    'Visceral_Pleural_Invasion':  '8.2',
    'AFP_Level':                  '8.1',
    'Liver_Fibrosis_Ishak':       '8.2',
    'Child_Pugh_Score':           '8.3',
    'CEA_Lab_Value':              '8.1',
    'MSI_MMR_Status':             '8.10',
    'RAS_Mutation':               '8.6',
}


def label_with_tcr_number(col: str) -> str:
    """Return column label prefixed with TCR field number if known.

    e.g. 'Age_at_Diagnosis' → '[2.1] Age_at_Diagnosis'
         'BMI' → 'BMI' (no TCR number — derived variable)
    """
    num = TCR_FIELD_NUMBER.get(col)
    return f'[{num}] {col}' if num else col


def apply_tcr_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with column names prefixed by TCR field numbers."""
    return df.rename(columns={c: label_with_tcr_number(c) for c in df.columns})


def generate_data_dictionary(df: pd.DataFrame) -> pd.DataFrame:
    """Generate a data dictionary DataFrame for the given clinical output.

    Args:
        df: The decoded clinical DataFrame (with or without derived variables)

    Returns:
        DataFrame with one row per column describing its metadata
    """
    rows = []
    for col in df.columns:
        meta = COLUMN_REGISTRY.get(col, ('unknown', 'unknown', ''))
        source, decoder, desc = meta[0], meta[1], meta[2]

        series = df[col]
        n_total = len(series)
        n_filled = series.dropna().apply(lambda x: str(x).strip() != '').sum()
        n_missing = n_total - n_filled
        pct_complete = round(100 * n_filled / n_total, 1)
        n_unique = series.dropna().nunique()

        # Determine data type
        dtype = str(series.dtype)
        if 'int' in dtype or 'float' in dtype:
            data_type = 'Numeric'
        elif 'bool' in dtype:
            data_type = 'Boolean'
        elif 'datetime' in dtype:
            data_type = 'Date'
        elif n_unique <= 20 and n_filled > 0:
            data_type = 'Categorical'
        elif series.dropna().astype(str).str.match(r'^\d{4}-\d{2}').any():
            data_type = 'Date (string)'
        else:
            data_type = 'Text'

        # Sample values
        uniq = series.dropna().unique()
        if len(uniq) <= 5:
            sample = ', '.join(str(v) for v in uniq[:5])
        else:
            sample = ', '.join(str(v) for v in uniq[:3]) + f' ... ({n_unique} unique)'

        rows.append({
            'Column': col,
            'Source_Field': source,
            'Decoder': decoder,
            'Description': desc,
            'Data_Type': data_type,
            'N_Filled': int(n_filled),
            'N_Missing': int(n_missing),
            'Completeness_%': pct_complete,
            'N_Unique': int(n_unique),
            'Sample_Values': sample[:100],
        })

    return pd.DataFrame(rows)
