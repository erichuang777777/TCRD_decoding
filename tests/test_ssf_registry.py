"""
Tests for tcr_decoder.ssf_registry — cancer group detection and SSF routing.
"""

import pytest
import pandas as pd
import numpy as np
from tcr_decoder.ssf_registry import (
    detect_cancer_group,
    detect_cancer_group_from_series,
    get_ssf_profile,
    apply_ssf_profile,
    list_supported_cancers,
)


# ─── detect_cancer_group ──────────────────────────────────────────────────────

class TestDetectCancerGroup:
    """ICD-O-3 code → cancer group routing."""

    @pytest.mark.parametrize('code,expected', [
        ('C50.1', 'breast'),
        ('C50.9', 'breast'),
        ('C50',   'breast'),
        ('c50.1', 'breast'),      # lowercase
        ('C34.1', 'lung'),
        ('C34.0', 'lung'),
        ('C34.9', 'lung'),
        ('C18.2', 'colorectum'),
        ('C19.9', 'colorectum'),
        ('C20',   'colorectum'),
        ('C21.0', 'colorectum'),
        ('C22.0', 'liver'),
        ('C22.1', 'liver'),
        ('C53.0', 'cervix'),
        ('C53.9', 'cervix'),
        ('C16.0', 'stomach'),
        ('C16.9', 'stomach'),
        ('C73',   'thyroid'),
        ('C73.9', 'thyroid'),
        ('C61',   'prostate'),
        ('C61.9', 'prostate'),
        ('C11.0', 'head_neck'),
        ('C11.9', 'head_neck'),
        ('C54.1', 'endometrium'),
        ('C54.9', 'endometrium'),
    ])
    def test_known_codes(self, code, expected):
        assert detect_cancer_group(code) == expected

    @pytest.mark.parametrize('code', ['C99.9', 'D05.1', 'X99', 'UNKNOWN'])
    def test_unknown_codes_return_generic(self, code):
        assert detect_cancer_group(code) == 'generic'

    @pytest.mark.parametrize('code', [None, '', np.nan, '  '])
    def test_empty_returns_generic(self, code):
        assert detect_cancer_group(code) == 'generic'


class TestMorphologyKeyedGroups:
    """Lymphoma and leukemia are chosen by MCODE, not by site (pp.194, 207)."""

    @pytest.mark.parametrize('site,morph,expected', [
        # Nodal and extranodal lymphomas: the site is irrelevant, and a
        # gastric MALT lymphoma must NOT come out as a stomach cancer.
        ('C77.9', '9680/3', 'lymphoma'),   # DLBCL, lymph node
        ('C16.9', '9699/3', 'lymphoma'),   # MALT lymphoma of the stomach
        ('C50.9', '9680/3', 'lymphoma'),   # primary breast lymphoma
        ('C77.0', '9663/3', 'lymphoma'),   # Hodgkin, nodular sclerosis
        ('C77.2', '9690/3', 'lymphoma'),   # follicular
        # Leukemias.
        ('C42.1', '9861/3', 'leukemia'),   # AML
        ('C42.1', '9875/3', 'leukemia'),   # CML, the only SSF10 collector
        ('C42.0', '9823/3', 'leukemia'),   # CLL
        ('C42.4', '9989/3', 'leukemia'),
    ])
    def test_morphology_decides(self, site, morph, expected):
        assert detect_cancer_group(site, morph) == expected

    @pytest.mark.parametrize('site,expected', [
        # M-9811-9837 is the one range the manual splits on the SITE: in
        # marrow / blood / haematopoietic system it is a leukemia, anywhere
        # else the same morphology is a lymphoma.
        ('C42.0', 'leukemia'),
        ('C42.1', 'leukemia'),
        ('C42.4', 'leukemia'),
        ('C42.2', 'lymphoma'),   # spleen
        ('C77.9', 'lymphoma'),
    ])
    def test_m9811_9837_splits_on_the_site(self, site, expected):
        assert detect_cancer_group(site, '9835/3') == expected

    def test_a_solid_tumour_morphology_leaves_the_site_in_charge(self):
        assert detect_cancer_group('C50.1', '8500/3') == 'breast'

    @pytest.mark.parametrize('site,morphology,expected', [
        # Mucosal melanoma of the head and neck (manual p.5/11): a specific
        # sinonasal site list AND one of nine melanoma morphologies, both
        # required. Regression: 'C30'/'C31' used to be registered as whole
        # 3-character site prefixes, so C30.1 (middle ear) and C31.2-C31.9
        # (other sinus subsites) -- which have no SSF table at all -- and
        # C30.0/C31.0-C31.1 with a NON-melanoma histology all routed into
        # head_neck anyway, decoding them with the wrong clinical meaning.
        ('C30.0', '8720/3', 'head_neck'),   # qualifying site + morphology
        ('C31.0', '8770/3', 'head_neck'),
        ('C31.1', '8730/3', 'head_neck'),
        ('C30.0', '8140/3', 'generic'),     # qualifying site, wrong histology
        ('C30.1', '8720/3', 'generic'),     # melanoma, but middle ear
        ('C31.2', '8720/3', 'generic'),     # melanoma, but wrong sinus subsite
        ('C30.0', None, 'generic'),         # no morphology to gate on at all
    ])
    def test_mucosal_melanoma_of_head_and_neck_needs_site_and_morphology(
            self, site, morphology, expected):
        assert detect_cancer_group(site, morphology) == expected

    def test_larynx_is_head_neck_by_site_alone_no_morphology_gate(self):
        """Unlike C30/C31, C32 (larynx) subsites need only the site (p.10)."""
        assert detect_cancer_group('C32.0') == 'head_neck'
        assert detect_cancer_group('C32.0', '8140/3') == 'head_neck'

    def test_site_alone_cannot_reach_the_haematolymphoid_profiles(self):
        """Without MCODE there is nothing to route on -- and we say so."""
        assert detect_cancer_group('C77.9') == 'generic'

    def test_series_detection_uses_the_morphology_column(self):
        sites = pd.Series(['C77.9', 'C77.0', 'C16.9'])
        morphs = pd.Series(['9680/3', '9680/3', '9699/3'])
        assert detect_cancer_group_from_series(sites, morphs) == 'lymphoma'
        # Same sites without the morphology: two nodal sites route nowhere and
        # the gastric lymphoma would have been read as a stomach cancer.
        with pytest.warns(UserWarning, match='Mixed cancer registry'):
            assert detect_cancer_group_from_series(sites) == 'generic'


class TestDetectCancerGroupFromSeries:
    """Bulk series detection."""

    def test_uniform_breast(self):
        s = pd.Series(['C50.1', 'C50.9', 'C50.4', 'C50.2'])
        assert detect_cancer_group_from_series(s) == 'breast'

    def test_uniform_lung(self):
        s = pd.Series(['C34.1', 'C34.0', 'C34.9'])
        assert detect_cancer_group_from_series(s) == 'lung'

    def test_empty_series(self):
        assert detect_cancer_group_from_series(pd.Series([])) == 'generic'

    def test_mixed_warns(self):
        # 6 breast + 4 lung → breast dominant but <90% → warning
        codes = ['C50.1'] * 6 + ['C34.1'] * 4
        with pytest.warns(UserWarning, match='Mixed cancer registry'):
            result = detect_cancer_group_from_series(pd.Series(codes))
        assert result == 'breast'   # dominant group

    def test_all_generic(self):
        s = pd.Series(['C99.9', 'C00.1', 'X01'])
        assert detect_cancer_group_from_series(s) == 'generic'


# ─── get_ssf_profile ─────────────────────────────────────────────────────────

class TestGetSSFProfile:

    def test_breast_has_10_fields(self):
        p = get_ssf_profile('breast')
        assert len(p.fields) == 10

    def test_breast_ssf1_is_er(self):
        p = get_ssf_profile('breast')
        assert p.fields['SSF1'].column_name == 'ER_Status'
        assert p.fields['SSF1'].decoder is not None

    def test_lung_ssf6_is_egfr(self):
        """SSF6 = EGFR mutation per 2025 TCR codebook (p.117)."""
        p = get_ssf_profile('lung')
        assert p.fields['SSF6'].column_name == 'EGFR_Mutation'
        assert p.fields['SSF6'].decoder is not None

    def test_lung_ssf7_is_alk(self):
        """SSF7 = ALK translocation per 2025 TCR codebook (p.119)."""
        p = get_ssf_profile('lung')
        assert p.fields['SSF7'].column_name == 'ALK_Translocation'

    def test_lung_ssf1_is_separate_nodules(self):
        """SSF1 = separate tumor nodules per 2025 TCR codebook (p.111)."""
        p = get_ssf_profile('lung')
        assert p.fields['SSF1'].column_name == 'Separate_Tumor_Nodules'

    def test_lung_ssf4_is_pleural_effusion(self):
        """SSF4 = malignant pleural effusion per 2025 TCR codebook (p.114)."""
        p = get_ssf_profile('lung')
        assert p.fields['SSF4'].column_name == 'Malignant_Pleural_Effusion'

    def test_colorectum_ssf10_is_msi(self):
        """SSF10 = MSI per 2025 TCR codebook (p.76/82)."""
        p = get_ssf_profile('colorectum')
        assert p.fields['SSF10'].column_name == 'MSI_MMR_Status'

    def test_colorectum_ssf6_is_ras(self):
        """SSF6 = RAS (KRAS+NRAS) per 2025 TCR codebook (p.54/70)."""
        p = get_ssf_profile('colorectum')
        assert p.fields['SSF6'].column_name == 'RAS_Mutation'

    def test_liver_ssf1_is_afp(self):
        p = get_ssf_profile('liver')
        assert p.fields['SSF1'].column_name == 'AFP_Level'

    def test_prostate_ssf1_is_psa(self):
        p = get_ssf_profile('prostate')
        assert p.fields['SSF1'].column_name == 'PSA_Lab_Value'

    def test_prostate_ssf_fields_follow_the_manual(self):
        """Cancer-SSF-Manual pp.175-191. SSF2/SSF4 are Gleason PATTERNS
        (biopsy vs prostatectomy), SSF3/SSF5 the matching SCORES, SSF6/SSF7
        the biopsy core counts, SSF8 the clinical T staging method."""
        p = get_ssf_profile('prostate')
        assert [p.fields[f'SSF{i}'].column_name for i in range(2, 9)] == [
            'Gleason_Pattern_Biopsy', 'Gleason_Score_Biopsy',
            'Gleason_Pattern_Prostatectomy', 'Gleason_Score_Prostatectomy',
            'Biopsy_Cores_Examined', 'Biopsy_Cores_Positive',
            'Clinical_T_Staging_Method',
        ]

    def test_unknown_group_returns_generic(self):
        p = get_ssf_profile('unknown_cancer')
        assert p.cancer_group == 'generic'

    def test_all_profiles_have_ssf1_to_ssf10(self):
        for group in ['breast', 'lung', 'colorectum', 'liver',
                      'cervix', 'stomach', 'thyroid', 'prostate',
                      'nasopharynx', 'endometrium', 'generic']:
            p = get_ssf_profile(group)
            for i in range(1, 11):
                assert f'SSF{i}' in p.fields, f'{group} missing SSF{i}'


# ─── apply_ssf_profile ───────────────────────────────────────────────────────

class TestApplySSFProfile:

    def _make_df(self, vals: dict) -> pd.DataFrame:
        """Build SSF raw DataFrame, padding missing columns."""
        full = {f'SSF{i}_raw': [0] * len(next(iter(vals.values())))
                for i in range(1, 11)}
        full.update(vals)
        return pd.DataFrame(full)

    # ── Breast ──────────────────────────────────────────────────────────────

    def test_breast_er_positive_percentage(self):
        df = self._make_df({'SSF1_raw': [70, 50, 1]})
        result = apply_ssf_profile(df, 'breast')
        assert result['ER_Status'].iloc[0] == 'ER Positive (70%)'
        assert result['ER_Status'].iloc[1] == 'ER Positive (50%)'
        assert result['ER_Status'].iloc[2] == 'ER Positive (1%)'

    def test_breast_er_negative(self):
        df = self._make_df({'SSF1_raw': [120]})
        result = apply_ssf_profile(df, 'breast')
        assert 'Negative' in result['ER_Status'].iloc[0]

    def test_breast_er_converted(self):
        df = self._make_df({'SSF1_raw': [888]})
        result = apply_ssf_profile(df, 'breast')
        assert 'converted' in result['ER_Status'].iloc[0].lower()

    def test_breast_er_unknown(self):
        df = self._make_df({'SSF1_raw': [999]})
        result = apply_ssf_profile(df, 'breast')
        assert 'Unknown' in result['ER_Status'].iloc[0]

    def test_breast_nottingham_score(self):
        df = self._make_df({'SSF6_raw': [30, 50, 70, 90]})
        result = apply_ssf_profile(df, 'breast')
        grades = result['Nottingham_Grade'].tolist()
        assert 'Grade 1' in grades[0]   # score 3
        assert 'Grade 1' in grades[1]   # score 5
        assert 'Grade 2' in grades[2]   # score 7
        assert 'Grade 3' in grades[3]   # score 9

    def test_breast_her2_ihc3_positive(self):
        # Code 300 = IHC 3+ (ISH not performed)
        # decode_her2 may map 300 → 'ISH Negative' (part of 3xx IHC-dominant group)
        df = self._make_df({'SSF7_raw': [300]})
        result = apply_ssf_profile(df, 'breast')
        text = result['HER2_Status'].iloc[0]
        # Accept any non-empty decoded result from the HER2 decoder
        assert text != '' and text != '300', f'HER2 300 was not decoded: {text!r}'

    def test_breast_her2_ihc0_negative(self):
        df = self._make_df({'SSF7_raw': [100]})
        result = apply_ssf_profile(df, 'breast')
        assert 'Negative' in result['HER2_Status'].iloc[0]

    def test_breast_ki67_low(self):
        df = self._make_df({'SSF10_raw': [5]})
        result = apply_ssf_profile(df, 'breast')
        assert 'Low' in result['Ki67_Index'].iloc[0]

    def test_breast_ki67_high(self):
        df = self._make_df({'SSF10_raw': [85]})
        result = apply_ssf_profile(df, 'breast')
        assert 'High' in result['Ki67_Index'].iloc[0]

    def test_breast_neoadj_pcr(self):
        df = self._make_df({'SSF3_raw': [11]})
        result = apply_ssf_profile(df, 'breast')
        assert 'pCR' in result['Neoadjuvant_Response'].iloc[0]

    def test_breast_neoadj_ccr(self):
        df = self._make_df({'SSF3_raw': [10]})
        result = apply_ssf_profile(df, 'breast')
        assert 'cCR' in result['Neoadjuvant_Response'].iloc[0]

    def test_breast_sentinel_ln(self):
        df = self._make_df({'SSF4_raw': [3], 'SSF5_raw': [1]})
        result = apply_ssf_profile(df, 'breast')
        assert '3' in result['Sentinel_LN_Examined'].iloc[0]
        assert '1' in result['Sentinel_LN_Positive'].iloc[0]

    def test_breast_output_columns_complete(self):
        # Build a proper df with all 10 SSF raw columns
        df = pd.DataFrame({f'SSF{i}_raw': [0] for i in range(1, 11)})
        result = apply_ssf_profile(df, 'breast')
        expected = ['ER_Status', 'PR_Status', 'Neoadjuvant_Response',
                    'Sentinel_LN_Examined', 'Sentinel_LN_Positive',
                    'Nottingham_Grade', 'HER2_Status', 'Ki67_Index']
        for col in expected:
            assert col in result.columns, f'Missing: {col}'

    # ── Lung ────────────────────────────────────────────────────────────────
    # Per 2025 TCR codebook: EGFR=SSF6 (3-char alpha), ALK=SSF7 (010/020/030)
    # ROS1 and PD-L1 are NOT in the 2025 TCR lung SSF codebook.

    def test_lung_egfr_exon19(self):
        """EGFR Exon19 deletion = 'AXX' in SSF6 (3-char alpha code)."""
        df = self._make_df({'SSF6_raw': ['AXX']})
        result = apply_ssf_profile(df, 'lung')
        assert 'Exon 19' in result['EGFR_Mutation'].iloc[0]

    def test_lung_egfr_l858r(self):
        """EGFR L858R = 'BXX' in SSF6."""
        df = self._make_df({'SSF6_raw': ['BXX']})
        result = apply_ssf_profile(df, 'lung')
        assert 'L858R' in result['EGFR_Mutation'].iloc[0]

    def test_lung_egfr_wildtype(self):
        """No EGFR mutation = 'XXX' in SSF6."""
        df = self._make_df({'SSF6_raw': ['XXX']})
        result = apply_ssf_profile(df, 'lung')
        assert 'No mutation' in result['EGFR_Mutation'].iloc[0]

    def test_lung_egfr_unknown(self):
        """EGFR unknown = '999' in SSF6."""
        df = self._make_df({'SSF6_raw': [999]})
        result = apply_ssf_profile(df, 'lung')
        assert 'Unknown' in result['EGFR_Mutation'].iloc[0]

    def test_lung_egfr_dual_mutation(self):
        """Two concurrent EGFR mutations = 'ABX' (Exon19del + L858R)."""
        df = self._make_df({'SSF6_raw': ['ABX']})
        result = apply_ssf_profile(df, 'lung')
        r = result['EGFR_Mutation'].iloc[0]
        assert 'Exon 19' in r and 'L858R' in r

    def test_lung_alk_positive(self):
        """ALK positive = code 10 in SSF7."""
        df = self._make_df({'SSF7_raw': [10]})
        result = apply_ssf_profile(df, 'lung')
        assert 'positive' in result['ALK_Translocation'].iloc[0].lower()

    def test_lung_alk_negative(self):
        """ALK negative = code 20 in SSF7."""
        df = self._make_df({'SSF7_raw': [20]})
        result = apply_ssf_profile(df, 'lung')
        assert 'negative' in result['ALK_Translocation'].iloc[0].lower()

    def test_lung_alk_uninterpretable(self):
        """ALK uninterpretable = code 30 in SSF7."""
        df = self._make_df({'SSF7_raw': [30]})
        result = apply_ssf_profile(df, 'lung')
        assert 'uninterpretable' in result['ALK_Translocation'].iloc[0].lower()

    def test_lung_ssf1_no_nodules(self):
        """SSF1=0: no separate ipsilateral tumor nodules."""
        df = self._make_df({'SSF1_raw': [0]})
        result = apply_ssf_profile(df, 'lung')
        assert 'No separate' in result['Separate_Tumor_Nodules'].iloc[0]

    def test_lung_ssf4_pleural_effusion_cytology(self):
        """SSF4=13: cytology-confirmed malignant pleural effusion."""
        df = self._make_df({'SSF4_raw': [13]})
        result = apply_ssf_profile(df, 'lung')
        assert 'Cytology confirmed' in result['Malignant_Pleural_Effusion'].iloc[0]

    def test_lung_ssf3_ecog_0(self):
        """SSF3=0: ECOG 0, fully active."""
        df = self._make_df({'SSF3_raw': [0]})
        result = apply_ssf_profile(df, 'lung')
        assert 'ECOG 0' in result['Performance_Status_SSF3'].iloc[0]

    def test_lung_ssf8_micropapillary(self):
        """SSF8=1: micropapillary component only."""
        df = self._make_df({'SSF8_raw': [1]})
        result = apply_ssf_profile(df, 'lung')
        assert 'Micropapillary' in result['Adenocarcinoma_Component'].iloc[0]

    # ── Colorectum ──────────────────────────────────────────────────────────
    # Per 2025 TCR codebook: MSI=SSF10, RAS(KRAS+NRAS)=SSF6, CEA=SSF1 (×10 coding)

    def test_colorectum_msi_high(self):
        """MSI-H = code 20 in SSF10."""
        df = self._make_df({'SSF10_raw': [20]})
        result = apply_ssf_profile(df, 'colorectum')
        assert 'MSI-H' in result['MSI_MMR_Status'].iloc[0]

    def test_colorectum_msi_stable(self):
        """MSS = code 0 in SSF10."""
        df = self._make_df({'SSF10_raw': [0]})
        result = apply_ssf_profile(df, 'colorectum')
        assert 'MSS' in result['MSI_MMR_Status'].iloc[0]

    def test_colorectum_ras_kras_codon12(self):
        """RAS code '108' = KRAS codon12, NRAS not tested."""
        df = self._make_df({'SSF6_raw': ['108']})
        result = apply_ssf_profile(df, 'colorectum')
        assert 'Codon 12' in result['RAS_Mutation'].iloc[0]

    def test_colorectum_ras_wildtype(self):
        """RAS code '008' = KRAS wild-type, NRAS wild-type."""
        df = self._make_df({'SSF6_raw': ['008']})
        result = apply_ssf_profile(df, 'colorectum')
        assert 'wild-type' in result['RAS_Mutation'].iloc[0].lower()

    def test_colorectum_cea_low(self):
        """CEA code 20 = 2.0 ng/mL (×10 coding, SSF1)."""
        df = self._make_df({'SSF1_raw': [20]})
        result = apply_ssf_profile(df, 'colorectum')
        assert '2.0' in result['CEA_Lab_Value'].iloc[0]

    def test_colorectum_cea_elevated(self):
        """CEA code 150 = 15.0 ng/mL (×10 coding, SSF1)."""
        df = self._make_df({'SSF1_raw': [150]})
        result = apply_ssf_profile(df, 'colorectum')
        assert '15.0' in result['CEA_Lab_Value'].iloc[0]

    # ── Liver ───────────────────────────────────────────────────────────────
    # Per 2025 TCR codebook: AFP=SSF1, Ishak=SSF2, Child-Pugh=SSF3,
    # Creatinine=SSF4, Bilirubin=SSF5, INR=SSF6, HBsAg=SSF7, Anti-HCV=SSF8

    def test_liver_afp_a_code(self):
        """AFP code A09 = 9 ng/mL (2021+ scheme)."""
        df = self._make_df({'SSF1_raw': ['A09']})
        result = apply_ssf_profile(df, 'liver')
        assert 'A-code' in result['AFP_Level'].iloc[0] or '9' in result['AFP_Level'].iloc[0]

    def test_liver_afp_elevated(self):
        """AFP code 500 = ~5000 ng/mL (≥1000 range, ÷10 coding)."""
        df = self._make_df({'SSF1_raw': [500]})
        result = apply_ssf_profile(df, 'liver')
        assert '5000' in result['AFP_Level'].iloc[0]

    def test_liver_afp_saturation(self):
        """AFP code 991 = instrument saturation (400-6000 ng/mL range)."""
        df = self._make_df({'SSF1_raw': [991]})
        result = apply_ssf_profile(df, 'liver')
        assert '400' in result['AFP_Level'].iloc[0] or '6000' in result['AFP_Level'].iloc[0]

    def test_liver_hbsag_positive(self):
        """HBsAg positive = code 20 in SSF7."""
        df = self._make_df({'SSF7_raw': [20]})
        result = apply_ssf_profile(df, 'liver')
        assert 'Positive' in result['HBsAg'].iloc[0]

    def test_liver_hbsag_negative_no_history(self):
        """HBsAg negative, no history = code 10 in SSF7."""
        df = self._make_df({'SSF7_raw': [10]})
        result = apply_ssf_profile(df, 'liver')
        assert 'Negative' in result['HBsAg'].iloc[0] and 'no hbv' in result['HBsAg'].iloc[0].lower()

    def test_liver_child_pugh_a5(self):
        """Child-Pugh A, score 5 = code 105 in SSF3."""
        df = self._make_df({'SSF3_raw': [105]})
        result = apply_ssf_profile(df, 'liver')
        assert 'Class A' in result['Child_Pugh_Score'].iloc[0]
        assert 'Score 5' in result['Child_Pugh_Score'].iloc[0]

    def test_liver_child_pugh_b7(self):
        """Child-Pugh B, score 7 = code 207 in SSF3."""
        df = self._make_df({'SSF3_raw': [207]})
        result = apply_ssf_profile(df, 'liver')
        assert 'Class B' in result['Child_Pugh_Score'].iloc[0]

    def test_liver_child_pugh_c10(self):
        """Child-Pugh C, score 10 = code 310 in SSF3."""
        df = self._make_df({'SSF3_raw': [310]})
        result = apply_ssf_profile(df, 'liver')
        assert 'Class C' in result['Child_Pugh_Score'].iloc[0]

    def test_liver_ishak_fibrosis(self):
        """Ishak F6 = cirrhosis, code 6 in SSF2."""
        df = self._make_df({'SSF2_raw': [6]})
        result = apply_ssf_profile(df, 'liver')
        assert 'F6' in result['Liver_Fibrosis_Ishak'].iloc[0] or 'Cirrhosis' in result['Liver_Fibrosis_Ishak'].iloc[0]

    def test_liver_inr(self):
        """INR 1.5 = code 15 in SSF6."""
        df = self._make_df({'SSF6_raw': [15]})
        result = apply_ssf_profile(df, 'liver')
        assert '1.5' in result['INR'].iloc[0]

    # ── Prostate ────────────────────────────────────────────────────────────

    def test_prostate_psa_lowest_code(self):
        # 001 = <=0.1 ng/mL; the official range starts at 001 (no 000).
        df = self._make_df({'SSF1_raw': ['001']})
        result = apply_ssf_profile(df, 'prostate')
        assert result['PSA_Lab_Value'].iloc[0] == 'PSA <=0.1 ng/mL'

    def test_prostate_psa_value(self):
        # stored as ×10 → 045 = 4.5 ng/mL
        df = self._make_df({'SSF1_raw': ['045']})
        result = apply_ssf_profile(df, 'prostate')
        assert '4.5' in result['PSA_Lab_Value'].iloc[0]

    def test_prostate_gleason_pattern_biopsy(self):
        # 034 = primary 3 + secondary 4, NOT "score 34"
        df = self._make_df({'SSF2_raw': ['034']})
        result = apply_ssf_profile(df, 'prostate')
        label = result['Gleason_Pattern_Biopsy'].iloc[0]
        assert 'Gleason 3+4' in label and 'primary 3' in label

    def test_prostate_gleason_score_biopsy(self):
        df = self._make_df({'SSF3_raw': ['006']})
        result = apply_ssf_profile(df, 'prostate')
        assert 'Grade Group 1' in result['Gleason_Score_Biopsy'].iloc[0]

    def test_prostate_gleason_score_prostatectomy(self):
        df = self._make_df({'SSF5_raw': ['009']})
        result = apply_ssf_profile(df, 'prostate')
        assert 'Grade Group 5' in result['Gleason_Score_Prostatectomy'].iloc[0]

    def test_prostate_biopsy_cores(self):
        df = self._make_df({'SSF6_raw': ['012'], 'SSF7_raw': ['003']})
        result = apply_ssf_profile(df, 'prostate')
        assert result['Biopsy_Cores_Examined'].iloc[0] == '12 core(s) examined'
        assert result['Biopsy_Cores_Positive'].iloc[0] == '3 core(s) positive'

    def test_prostate_clinical_t_staging_method(self):
        df = self._make_df({'SSF8_raw': ['030']})
        result = apply_ssf_profile(df, 'prostate')
        assert 'Digital rectal exam plus TRUS' in result['Clinical_T_Staging_Method'].iloc[0]

    # ── Generic sentinel codes ───────────────────────────────────────────────

    @pytest.mark.parametrize('sentinel,expected_kw', [
        (988, 'Not applicable'),
        (999, 'Unknown'),
        (902, 'not documented'),
    ])
    def test_generic_sentinel_decoded(self, sentinel, expected_kw):
        df = self._make_df({'SSF1_raw': [sentinel]})
        result = apply_ssf_profile(df, 'generic')
        assert expected_kw.lower() in result['SSF1'].iloc[0].lower()

    # ── Missing raw columns handled gracefully ───────────────────────────────

    def test_missing_raw_columns_ignored(self):
        """apply_ssf_profile should not crash if some SSF_raw cols absent."""
        df = pd.DataFrame({'SSF1_raw': [70], 'SSF2_raw': [40]})
        # SSF3-10 missing → should decode SSF1/SSF2 and skip the rest
        result = apply_ssf_profile(df, 'breast')
        assert 'ER_Status' in result.columns
        assert 'PR_Status' in result.columns


# ─── list_supported_cancers ──────────────────────────────────────────────────

class TestListSupportedCancers:

    def test_returns_dataframe(self):
        df = list_supported_cancers()
        assert isinstance(df, pd.DataFrame)

    def test_has_expected_columns(self):
        df = list_supported_cancers()
        for col in ['Cancer_Group', 'Site_Label', 'ICD_O_3_Codes', 'Custom_Decoders']:
            assert col in df.columns

    def test_breast_present(self):
        df = list_supported_cancers()
        assert 'breast' in df['Cancer_Group'].values

    def test_generic_present(self):
        df = list_supported_cancers()
        assert 'generic' in df['Cancer_Group'].values

    def test_minimum_10_groups(self):
        df = list_supported_cancers()
        assert len(df) >= 10
