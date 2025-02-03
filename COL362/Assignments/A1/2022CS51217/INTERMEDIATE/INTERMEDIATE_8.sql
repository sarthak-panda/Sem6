%%sql select subject_id,AVG(TO_TIMESTAMP(dischtime, 'YYYY-MM-DD HH24:MI:SS')-TO_TIMESTAMP(admittime, 'YYYY-MM-DD HH24:MI:SS')) as avg_duration
from hosp.admissions
where dischtime is not null
group by subject_id
order by subject_id