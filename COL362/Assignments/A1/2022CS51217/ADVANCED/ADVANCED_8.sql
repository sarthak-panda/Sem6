
with T0 as (
	with T0_0 as (
		select distinct subject_id,hadm_id
		from hosp.diagnoses_icd
		where icd_code LIKE 'I10%'
	)
	,T0_1 as (
		select distinct subject_id,hadm_id
		from hosp.diagnoses_icd
		where icd_code LIKE 'I50%'
	)
	,T0_2 as (
		select subject_id, hadm_id, LEAD(hadm_id) OVER (PARTITION BY subject_id ORDER BY admittime) AS next_hadm_id
		from hosp.admissions	
	)

	select distinct T0_0.subject_id
	from T0_0 left outer join T0_2 on T0_0.subject_id=T0_2.subject_id and T0_0.hadm_id=T0_2.hadm_id
	where T0_0.hadm_id = ANY (select hadm_id from T0_1) or (T0_2.next_hadm_id is not null and T0_2.next_hadm_id = ANY (select hadm_id from T0_1))
)
,T1 as(
	select subject_id,array_agg(distinct hadm_id) as admissionsI10
	from hosp.diagnoses_icd
	where icd_code LIKE 'I10%'
	group by subject_id
	order by subject_id
),
T2 as (
	select subject_id,array_agg(distinct hadm_id) as admissionsI50
	from hosp.diagnoses_icd
	where icd_code LIKE 'I50%'
	group by subject_id
	order by subject_id
),
T3 as (
	select subject_id,hadm_id
	from hosp.admissions
	order by subject_id,admittime
), 
T4 as (
	select subject_id,array_agg(hadm_id) as admissions_seq
	from T3
	group by subject_id
)
--T5 as (
--	select T1.subject_id,T1.admissionsI10,T2.admissionsI50,T4.admissions_seq
--	from T1 join T2 on T1.subject_id=T2.subject_id join T4 on T1.subject_id=T4.subject_id
--	where array_length(admissions_seq,1)>=4
--)
,T5 AS (
    SELECT 
        T1.subject_id,
        T1.admissionsI10,
        T2.admissionsI50,
        T4.admissions_seq,
        
        -- Find first element in admissions_seq that exists in admissionsI10
        (SELECT seq 
         FROM unnest(T4.admissions_seq) AS seq 
         WHERE seq = ANY (T1.admissionsI10) 
         LIMIT 1) AS first_I10,

        -- Find last element in admissions_seq that exists in admissionsI50
        (SELECT seq 
         FROM (SELECT seq, array_position(T4.admissions_seq, seq) AS pos
               FROM unnest(T4.admissions_seq) AS seq) AS ordered_seq
         WHERE seq = ANY (T2.admissionsI50)
         ORDER BY pos DESC
         LIMIT 1) AS last_I50
         
    FROM T1
    JOIN T2 ON T1.subject_id = T2.subject_id
    JOIN T4 ON T1.subject_id = T4.subject_id
    WHERE array_length(T4.admissions_seq, 1) >= 4
)
,T6 as (
SELECT *,
       -- Extract subarray between first_I10 and last_I50
       admissions_seq[array_position(admissions_seq, first_I10)+1:
                      array_position(admissions_seq, last_I50)-1] AS subArrFL
FROM T5
)
,T7 as (
    select subject_id,unnest(subArrFL) as hadm_id_req
    from T6
    where array_length(subArrFL,1)>=4
)
select distinct subject_id,hadm_id as admission_id,drug
from hosp.prescriptions
where ((subject_id,hadm_id) in (select subject_id,hadm_id_req from T7)) and subject_id in (select subject_id from T0)
order by subject_id,hadm_id,drug
