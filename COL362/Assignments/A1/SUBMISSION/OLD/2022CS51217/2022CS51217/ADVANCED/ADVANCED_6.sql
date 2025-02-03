
with 
T1_0 as (
	select subject_id,stay_id,itemid
	from icu.inputevents
	group by subject_id,stay_id,itemid	
)
,T1 as (
	select subject_id,stay_id,itemid,sum(amount) as input_vol
	from icu.inputevents
	where amountuom='ml'
	group by subject_id,stay_id,itemid
)
,T1_1 as (
	select subject_id,stay_id,sum(input_vol) as total_input_vol
	from T1
	group by subject_id,stay_id
)
,T2_0 as (
	select subject_id,stay_id,itemid,sum(value) as output_vol
	from icu.outputevents
	group by subject_id,stay_id,itemid	
)
,T2 as (
	select subject_id,stay_id,itemid,sum(value) as output_vol
	from icu.outputevents
	where valueuom='ml'
	group by subject_id,stay_id,itemid
)
,T2_1 as (
	select subject_id,stay_id,sum(output_vol) as total_output_vol
	from T2
	group by subject_id,stay_id
)
,T3 as (
	select itemid,label
	from icu.d_items
)
,T4 as (
	select distinct T1_1.subject_id,T1_1.stay_id
	from T1_1 full outer join T2_1 on T1_1.subject_id=T2_1.subject_id and T1_1.stay_id=T2_1.stay_id
	where ABS(COALESCE(T1_1.total_input_vol,0) - COALESCE(T2_1.total_output_vol, 0)) > 2000
	group by T1_1.subject_id,T1_1.stay_id
	order by T1_1.subject_id,T1_1.stay_id
)
,T5 as (
	select subject_id,stay_id,T1_0.itemid,'input' as input_or_output,T3.label
	from T1_0 join T3 on T1_0.itemid=T3.itemid
	where (subject_id,stay_id) in (select subject_id,stay_id from T4)
	UNION
	select subject_id,stay_id,T2_0.itemid,'output' as input_or_output,T3.label
	from T2_0 join T3 on T2_0.itemid=T3.itemid
	where (subject_id,stay_id) in (select subject_id,stay_id from T4)
)
select subject_id,stay_id,itemid as item_id,input_or_output,label as description
from T5
order by subject_id,stay_id,item_id,input_or_output