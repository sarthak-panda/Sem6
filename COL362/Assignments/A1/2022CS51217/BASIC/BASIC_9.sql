select icd_code,icd_version
from (
	select distinct icd_code,icd_version /*here distinct needed*/
	from hosp.diagnoses_icd
	intersect
	select distinct icd_code,icd_version /*here distinct needed*/
	from hosp.procedures_icd
) as a
order by icd_code,icd_version