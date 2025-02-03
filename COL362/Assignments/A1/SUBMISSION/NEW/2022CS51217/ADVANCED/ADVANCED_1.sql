WITH icd_sets AS (
    select hosp.admissions.subject_id, hosp.admissions.hadm_id, COALESCE(array_agg(DISTINCT icd_code ORDER BY icd_code), ARRAY[]::text[]) AS icd_code_set
    from hosp.admissions
    left join hosp.diagnoses_icd on hosp.admissions.hadm_id = hosp.diagnoses_icd.hadm_id
    /*SELECT
        subject_id,
        hadm_id,
        array_agg(DISTINCT icd_code ORDER BY icd_code) AS icd_code_set
    FROM hosp.diagnoses_icd*/
    GROUP BY hosp.admissions.subject_id, hosp.admissions.hadm_id
    
),
distinct_icd_sets AS (
    SELECT subject_id, icd_code_set
    FROM icd_sets
    GROUP BY subject_id, icd_code_set
),
drug_sets AS (
    select hosp.admissions.subject_id, hosp.admissions.hadm_id, COALESCE(array_agg(DISTINCT drug ORDER BY drug), ARRAY[]::text[]) AS drug_set
    from hosp.admissions
    left join hosp.prescriptions on hosp.admissions.hadm_id = hosp.prescriptions.hadm_id
    /*SELECT
        subject_id,
        hadm_id,
        array_agg(DISTINCT drug ORDER BY drug) AS drug_set
    FROM hosp.prescriptions*/
    GROUP BY hosp.admissions.subject_id, hosp.admissions.hadm_id
),
distinct_drug_sets AS (
    SELECT subject_id, drug_set
    FROM drug_sets
    GROUP BY subject_id, drug_set
),
count_distinct_icd_sets AS (
    SELECT subject_id, COUNT(*) AS distinct_icd_sets_count
    FROM distinct_icd_sets
    GROUP BY subject_id
),
count_distinct_drug_sets AS (
    SELECT subject_id, COUNT(*) AS distinct_drug_count
    FROM distinct_drug_sets
    GROUP BY subject_id
),
count_total_admissions AS (
    SELECT subject_id, COUNT(*) AS total_admissions
    FROM hosp.admissions
    GROUP BY subject_id
)
SELECT count_total_admissions.subject_id,total_admissions,distinct_icd_sets_count as num_distinct_diagnoses_set_count,distinct_drug_count as num_distinct_medications_set_count
FROM count_distinct_icd_sets JOIN count_distinct_drug_sets ON count_distinct_icd_sets.subject_id = count_distinct_drug_sets.subject_id JOIN count_total_admissions ON count_distinct_icd_sets.subject_id = count_total_admissions.subject_id
WHERE distinct_icd_sets_count >= 3 OR distinct_drug_count >= 3
ORDER BY total_admissions DESC, num_distinct_diagnoses_set_count DESC, subject_id ASC