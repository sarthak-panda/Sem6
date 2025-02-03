with cte as (
	select subject_id,avg(result_value::NUMERIC) as avg_BMI
	from hosp.omr
	where result_name='BMI (kg/m2)' and subject_id in (
		select subject_id
		from hosp.emar
		where medication LIKE '%OxyCODONE (Immediate Release)%'
		INTERSECT
		select subject_id
		from hosp.emar
		where medication LIKE '%Insulin%'
	)
	group by subject_id
)
select ROUND(avg(avg_BMI),10) as avg_BMI
from cte