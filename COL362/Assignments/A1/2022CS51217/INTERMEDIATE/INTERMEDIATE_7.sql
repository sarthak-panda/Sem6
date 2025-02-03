with count as (
	with earliest as (
		select distinct on (subject_id) subject_id,hadm_id as earliest_hadm_id
		from hosp.admissions
		order by subject_id,admittime
	)
	, latest as (
		select distinct on (subject_id) subject_id,hadm_id as latest_hadm_id
		from hosp.admissions
		order by subject_id,admittime desc
	)
	select gender,count(*) as cnt
	from hosp.patients join earliest on hosp.patients.subject_id=earliest.subject_id join latest on hosp.patients.subject_id=latest.subject_id
	where exists (
		select distinct icd_code,icd_version
		from hosp.diagnoses_icd
		where hadm_id=earliest_hadm_id
		intersect
		select distinct icd_code,icd_version
		from hosp.diagnoses_icd
		where hadm_id=latest_hadm_id
	)
	group by gender
)
,total as (
	select gender,count(*) as tot
	from hosp.patients
	group by gender
)
,total_1 as (
	select sum(cnt) as tot
	from count
)
select count.gender,ROUND((cnt::NUMERIC/tot::NUMERIC)*100,2) as percentage
from count,total_1
order by percentage desc