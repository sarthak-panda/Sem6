select count(distinct subject_id), substring(admittime,1,4) as year
from hosp.admissions
group by year
order by count desc,year
limit 5;