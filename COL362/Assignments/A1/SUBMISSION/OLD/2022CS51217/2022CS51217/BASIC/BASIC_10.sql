select distinct hcpcs_cd,short_description
from hosp.hcpcsevents
where short_description ILIKE '%Hospital observation%'-- short description has 'Hospital observation' as substring (case-insensitive)
order by hcpcs_cd,short_description