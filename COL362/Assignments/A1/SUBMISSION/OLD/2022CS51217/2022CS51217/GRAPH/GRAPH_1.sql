
with T1 as (
	select subject_id,hadm_id,admittime,dischtime
	from hosp.admissions
	order by admittime
	limit 200
)
select distinct x.subject_id as subject_id1,y.subject_id as subject_id2
from T1 x, T1 y
where x.subject_id<y.subject_id and ((x.admittime::TIMESTAMP > y.admittime::TIMESTAMP and x.admittime::TIMESTAMP < y.dischtime::TIMESTAMP) or (y.admittime::TIMESTAMP > x.admittime::TIMESTAMP and y.admittime::TIMESTAMP < x.dischtime::TIMESTAMP)) and 
exists (
	select icd_code,icd_version
	from hosp.diagnoses_icd
	where hadm_id=x.hadm_id
	INTERSECT
	select icd_code,icd_version
	from hosp.diagnoses_icd
	where hadm_id=y.hadm_id
)
order by x.subject_id,y.subject_id