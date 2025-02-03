
WITH T1 AS (
    SELECT subject_id, hadm_id, 
        CASE 
            WHEN array_length(array_agg(distinct drug), 1) = 2 THEN 'both'
            WHEN lower(array_to_string(array_agg(distinct drug), ',')) LIKE '%amlodipine%' THEN 'amlodipine'
            WHEN lower(array_to_string(array_agg(distinct drug), ',')) LIKE '%lisinopril%' THEN 'lisinopril'
        END AS drug_category
    FROM hosp.prescriptions
    WHERE drug ILIKE '%amlodipine%' OR drug ILIKE '%lisinopril%'
    GROUP BY subject_id, hadm_id
)
/*, service_paths AS (
    WITH RECURSIVE service_patch_fetcher AS (
        -- Base case: Get the first service transition
        SELECT subject_id, hadm_id, prev_service, curr_service, ARRAY[curr_service] AS path_history
        FROM hosp.services
        WHERE prev_service IS NULL

        UNION ALL

        -- Recursive case: Get the next transition, stopping when a cycle is detected
        SELECT hs.subject_id, hs.hadm_id, hs.prev_service, hs.curr_service, spf.path_history || hs.curr_service
        FROM hosp.services hs
        JOIN service_patch_fetcher spf 
        ON hs.subject_id = spf.subject_id
        AND hs.hadm_id = spf.hadm_id
        AND hs.prev_service = spf.curr_service
        WHERE NOT hs.curr_service = ANY(spf.path_history)  -- Stop if service is already in path
    )
    SELECT subject_id, hadm_id, curr_service AS path
    FROM service_patch_fetcher
)*/
, service_paths AS (
    -- Get the sequence of services for the second admission
    SELECT r.subject_id, r.hadm_id, 
            STRING_AGG(s.curr_service, ',' ORDER BY s.transfertime) AS service_path
    FROM T1 r
    JOIN hosp.services s 
        ON r.subject_id = s.subject_id 
        AND r.hadm_id = s.hadm_id
    GROUP BY r.subject_id, r.hadm_id
)
SELECT T1.subject_id, T1.hadm_id, T1.drug_category as drug, string_to_array(service_paths.service_path,',') as services
FROM T1
LEFT JOIN service_paths 
    ON T1.subject_id = service_paths.subject_id 
    AND T1.hadm_id = service_paths.hadm_id
--GROUP BY T1.subject_id, T1.hadm_id, T1.drug_category
order by subject_id,hadm_id