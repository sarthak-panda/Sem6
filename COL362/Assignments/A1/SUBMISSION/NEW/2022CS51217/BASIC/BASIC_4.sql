select enter_provider_id,count(distinct medication) as count
from hosp.emar
where enter_provider_id is not null
group by enter_provider_id
order by count desc