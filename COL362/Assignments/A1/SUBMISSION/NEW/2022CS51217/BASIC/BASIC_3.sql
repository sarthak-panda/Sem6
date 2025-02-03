select hadm_id,gender,TO_TIMESTAMP(dischtime, 'YYYY-MM-DD HH24:MI:SS')-TO_TIMESTAMP(admittime, 'YYYY-MM-DD HH24:MI:SS') as duration
from hosp.admissions join hosp.patients on hosp.admissions.subject_id=hosp.patients.subject_id--why the hell i joined the tables, check the question again...
where dischtime is not null
order by duration,hadm_id