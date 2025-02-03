select o.subject_id,o.hadm_id,i.dod
from hosp.admissions as o
join(
	select hosp.patients.subject_id,min(hosp.admissions.admittime) as earliest_admittime,hosp.patients.dod
	from hosp.patients
	join hosp.admissions on hosp.patients.subject_id=hosp.admissions.subject_id
	where dod is not null
	group by hosp.patients.subject_id,hosp.patients.dod
) as i
on o.subject_id=i.subject_id and o.admittime=i.earliest_admittime
order by o.subject_id