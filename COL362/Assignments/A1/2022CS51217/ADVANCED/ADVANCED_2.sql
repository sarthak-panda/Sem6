with T1 as(
	select subject_id,hadm_id,count(distinct micro_specimen_id) as resistant_antibiotic_count
	from hosp.microbiologyevents
	where interpretation='R' and hadm_id is not null
	group by subject_id,hadm_id
	having count(distinct micro_specimen_id)>=2
	order by subject_id
),T2 as (
	select subject_id,hadm_id,ROUND(EXTRACT(EPOCH FROM (TO_TIMESTAMP(outtime, 'YYYY-MM-DD HH24:MI:SS') - TO_TIMESTAMP(intime, 'YYYY-MM-DD HH24:MI:SS'))) / 3600, 2) as icu_length_of_stay_hours
	from icu.icustays
	where outtime is not null	
),T3 as (
	select subject_id,hadm_id,discharge_location
	from hosp.admissions
	where discharge_location='DIED'
)
select T1.subject_id,T1.hadm_id,resistant_antibiotic_count,COALESCE(icu_length_of_stay_hours,0) as icu_length_of_stay_hours,case when discharge_location = 'DIED' then 1 else 0 end as died_in_hospital
from T1 left outer join T2 on T1.subject_id=T2.subject_id and T1.hadm_id=T2.hadm_id left outer join T3 on T1.subject_id=T3.subject_id and T1.hadm_id=T3.hadm_id
order by died_in_hospital desc,resistant_antibiotic_count desc,icu_length_of_stay_hours desc,subject_id,hadm_id
