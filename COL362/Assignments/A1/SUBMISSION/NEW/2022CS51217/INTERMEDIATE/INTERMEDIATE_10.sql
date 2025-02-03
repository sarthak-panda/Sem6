with final as (
	with temp as (
		select subject_id
		from hosp.admissions
		group by subject_id
		having count(*)>1
	),
	first_admission as (
		select distinct on (subject_id) subject_id,hadm_id,admittime as first_admittime
		from hosp.admissions
		order by subject_id,admittime
	)
	select distinct first_admission.subject_id,first_admission.hadm_id,first_admittime
	from hosp.d_icd_diagnoses join hosp.diagnoses_icd on hosp.d_icd_diagnoses.icd_code=hosp.diagnoses_icd.icd_code and hosp.d_icd_diagnoses.icd_version=hosp.diagnoses_icd.icd_version join first_admission on hosp.diagnoses_icd.hadm_id=first_admission.hadm_id
	where long_title ILIKE '%kidney%' and first_admission.subject_id in (select subject_id from temp)
	order by first_admittime desc
	limit 100
)
select subject_id
from final
order by subject_id