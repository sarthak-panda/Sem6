select justify_hours(AVG(dischtime::TIMESTAMP-admittime::TIMESTAMP)) as avg_duration --date trunc was used to remove  decimal precision in seconds
from hosp.admissions --join hosp.patients on hosp.admissions.subject_id=hosp.patients.subject_id
where dischtime is not null and 
hosp.admissions.hadm_id in (
	select hadm_id
	from hosp.diagnoses_icd
	where icd_code='4019' and icd_version='9'
)