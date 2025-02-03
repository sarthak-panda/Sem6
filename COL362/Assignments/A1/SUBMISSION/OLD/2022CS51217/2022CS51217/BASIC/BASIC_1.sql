select subject_id
from hosp.patients
where anchor_age>89 and gender='F'
order by subject_id