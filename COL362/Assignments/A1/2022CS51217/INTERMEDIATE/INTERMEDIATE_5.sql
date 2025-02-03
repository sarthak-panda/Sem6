with proc_t as (
	select subject_id,hadm_id,count(*) as count_procedures
	from hosp.procedures_icd
	group by subject_id,hadm_id
)
, diag_t as (
	select subject_id,hadm_id,count(*) as count_diagnoses
	from hosp.diagnoses_icd
	group by subject_id,hadm_id
)
select hosp.admissions.subject_id,hosp.admissions.hadm_id,count_procedures,count_diagnoses
from hosp.admissions left outer join proc_t on hosp.admissions.hadm_id=proc_t.hadm_id left outer join diag_t on hosp.admissions.hadm_id=diag_t.hadm_id
where admission_type='URGENT' and hospital_expire_flag=1
order by subject_id,hadm_id,count_procedures desc,count_diagnoses desc