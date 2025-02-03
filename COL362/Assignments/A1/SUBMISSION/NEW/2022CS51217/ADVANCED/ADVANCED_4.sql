with outer_t as (
	with temp as (
		select subject_id,hadm_id,transfer_id
		from hosp.transfers
		where eventtype='transfer'
		order by subject_id,hadm_id,intime
	)
	select subject_id,hadm_id,array_agg(transfer_id) AS transfers,array_length(array_agg(transfer_id),1) as transfer_count
	from temp
	group by subject_id,hadm_id
)
select subject_id,hadm_id,transfers
from outer_t
where transfer_count>=(SELECT MAX(transfer_count) FROM outer_t)
order by transfer_count,hadm_id,subject_id