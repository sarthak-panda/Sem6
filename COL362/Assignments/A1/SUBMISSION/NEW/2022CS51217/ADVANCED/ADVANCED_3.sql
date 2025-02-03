with T1 as (
	select subject_id,hadm_id,count(distinct icd_code) as count_procedures
	from hosp.procedures_icd
	group by subject_id,hadm_id
	having count(distinct icd_code)>1	
), T1_1 as (
	select distinct subject_id
	from T1
), T1_2 as (
	select subject_id,count(distinct icd_code) as count_procedures_overall
	from hosp.procedures_icd
	group by subject_id
)
, T2 as (
	select subject_id
	from T1_1
	INTERSECT
	select subject_id
	from hosp.diagnoses_icd
	where icd_code LIKE 'T81%'
	group by subject_id
), T3 as (
	select subject_id,hadm_id,count(distinct transfer_id) as count_transfers
	from hosp.transfers
	where subject_id in (select subject_id from T2) and hadm_id is not null
	group by subject_id,hadm_id
	order by subject_id,hadm_id
), T4 as (
	select subject_id,AVG(count_transfers) as avg_transfers_per_admission_for_each_patient
	from T3
	group by subject_id
), T5 as (
	select AVG(avg_transfers_per_admission_for_each_patient) as overall_avg
	from T4
)
select T4.subject_id,count_procedures_overall as distinct_procedures_count,ROUND(avg_transfers_per_admission_for_each_patient::NUMERIC,2) as average_transfers
from T4 join T1_2 on T4.subject_id=T1_2.subject_id
where avg_transfers_per_admission_for_each_patient>=(select overall_avg from T5)
order by average_transfers desc,distinct_procedures_count desc,T4.subject_id
