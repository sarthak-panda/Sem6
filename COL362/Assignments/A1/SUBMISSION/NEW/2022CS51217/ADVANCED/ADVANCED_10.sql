
WITH T0 AS (
    select distinct on (subject_id) subject_id,hadm_id
    from hosp.admissions
    order by subject_id,admittime
)
,T1 AS (
    -- Get patients diagnosed with an ICD code starting with "I2"
    SELECT DISTINCT subject_id, hadm_id
    FROM hosp.diagnoses_icd
    WHERE icd_code LIKE 'I2%' AND (subject_id,hadm_id) in (select subject_id,hadm_id from T0)
)
, T2 AS (
    -- Get first admission with I2 diagnosis
    SELECT subject_id, hadm_id, admittime::TIMESTAMP AS admittime, dischtime::TIMESTAMP AS dischtime
    FROM hosp.admissions
    WHERE (subject_id, hadm_id) IN (SELECT subject_id, hadm_id FROM T1)
)
, Readmissions AS (
    -- Get the first readmission within 180 days after the first discharge
    SELECT DISTINCT ON (t2.subject_id)  -- Select only the first readmission per patient
        t2.subject_id, 
        t2.hadm_id AS first_hadm_id,
        t2.dischtime AS first_discharge_time,
        t3.hadm_id AS second_hadm_id,
        t3.admittime::TIMESTAMP AS second_admit_time,  -- Cast to TIMESTAMP
        t3.dischtime::TIMESTAMP AS second_discharge_time,  -- Cast to TIMESTAMP
        TO_CHAR(AGE(t3.admittime::TIMESTAMP,t2.dischtime::TIMESTAMP), 'YYYY-MM-DD HH24:MI:SS') AS time_gap  -- Ensure both are TIMESTAMP for subtraction
    FROM T2 t2
    JOIN hosp.admissions t3 
        ON t2.subject_id = t3.subject_id
        AND t3.admittime::TIMESTAMP > t2.dischtime::TIMESTAMP  -- Ensure valid readmission
        AND t3.admittime::TIMESTAMP <= (t2.dischtime::TIMESTAMP + INTERVAL '180 days')  -- Within 180 days
    ORDER BY t2.subject_id, t3.admittime  -- Earliest readmission first
)
, ServiceList AS (
    -- Get the sequence of services for the second admission
    SELECT r.subject_id, r.second_hadm_id, 
            STRING_AGG(s.curr_service, ',' ORDER BY s.transfertime) AS service_path
    FROM Readmissions r
    JOIN hosp.services s 
        ON r.subject_id = s.subject_id 
        AND r.second_hadm_id = s.hadm_id
    GROUP BY r.subject_id, r.second_hadm_id
)
SELECT 
    r.subject_id,
    --r.first_hadm_id,
    --r.first_discharge_time,
    r.second_hadm_id,
    --r.second_admit_time,
    --r.second_discharge_time,
    r.time_gap as time_gap_between_admissions,
    COALESCE(string_to_array(s.service_path, ','), ARRAY[]::text[]) AS services
FROM Readmissions r
LEFT JOIN ServiceList s 
    ON r.subject_id = s.subject_id 
    AND r.second_hadm_id = s.second_hadm_id
ORDER BY array_length(string_to_array(s.service_path, ','), 1) desc,time_gap_between_admissions desc, r.subject_id, r.second_admit_time;
