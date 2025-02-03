with temp as (
	select subject_id,pharmacy_id,count(*) as count
	from hosp.prescriptions
	group by subject_id,pharmacy_id
	having count(*)>1
	order by count desc,subject_id,pharmacy_id
)
select subject_id,pharmacy_id
from temp