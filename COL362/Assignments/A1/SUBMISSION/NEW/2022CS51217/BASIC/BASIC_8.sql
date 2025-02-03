select pharmacy_id
from (
	select distinct pharmacy_id /*here distinct not needed*/
	from hosp.pharmacy
	except
	select distinct pharmacy_id /*here distinct needed*/
	from hosp.prescriptions
) as a
order by pharmacy_id