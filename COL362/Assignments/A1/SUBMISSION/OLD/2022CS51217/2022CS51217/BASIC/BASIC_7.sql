select hosp.patients.subject_id,max(hadm_id) as latest_hadm_id,dod
from hosp.patients join hosp.admissions on hosp.patients.subject_id=hosp.admissions.subject_id
where dod is not null
group by hosp.patients.subject_id,dod
order by subject_id