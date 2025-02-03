select count(distinct hosp.admissions.hadm_id)
from hosp.admissions join hosp.emar on hosp.admissions.hadm_id=hosp.emar.hadm_id join hosp.emar_detail on hosp.emar.emar_id=hosp.emar_detail.emar_id
where reason_for_no_barcode='Barcode Damaged' and marital_status<>'MARRIED'--to check about null values