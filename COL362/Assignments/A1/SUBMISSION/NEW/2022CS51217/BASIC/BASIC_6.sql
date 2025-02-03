select hosp.patients.subject_id,count(*)
from icu.icustays right outer join hosp.patients on icu.icustays.subject_id=hosp.patients.subject_id
group by hosp.patients.subject_id
order by count,hosp.patients.subject_id