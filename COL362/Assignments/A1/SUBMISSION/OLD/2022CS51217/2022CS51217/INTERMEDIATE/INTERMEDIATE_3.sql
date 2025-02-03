with pe as (
	select caregiver_id,count(*)
	from icu.procedureevents
	where caregiver_id is not null
	group by caregiver_id
)
,ce as (
	select caregiver_id,count(*)
	from icu.chartevents
	where caregiver_id is not null
	group by caregiver_id
)
,dte as (
	select caregiver_id,count(*)
	from icu.datetimeevents
	where caregiver_id is not null
	group by caregiver_id
)
select icu.caregiver.caregiver_id,COALESCE(pe.count,0) as procedureevents_count,COALESCE(ce.count,0) as chartevents_count,COALESCE(dte.count,0) as datetimeevents_count
from icu.caregiver left outer join pe on icu.caregiver.caregiver_id=pe.caregiver_id left outer join dte on icu.caregiver.caregiver_id=dte.caregiver_id left outer join ce on icu.caregiver.caregiver_id=ce.caregiver_id
order by icu.caregiver.caregiver_id,procedureevents_count,chartevents_count,datetimeevents_count