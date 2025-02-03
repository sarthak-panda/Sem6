
with T1 as (
	select subject_id,hadm_id,admittime,dischtime
	from hosp.admissions
	order by admittime
	limit 200
)
,T2 as (
select distinct x.subject_id as subject_id1,y.subject_id as subject_id2
from T1 x, T1 y
where ((x.admittime::TIMESTAMP > y.admittime::TIMESTAMP and x.admittime::TIMESTAMP < y.dischtime::TIMESTAMP) or (y.admittime::TIMESTAMP > x.admittime::TIMESTAMP and y.admittime::TIMESTAMP < x.dischtime::TIMESTAMP))
)
SELECT
	CASE 
		WHEN EXISTS (
			SELECT 1 FROM T2 
			WHERE subject_id1 = '10006580' AND subject_id2 = '10003400'
		) THEN 1
		ELSE 0
	END AS path_exists