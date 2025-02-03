select hosp.diagnoses_icd.subject_id,count(distinct hosp.admissions.hadm_id) as count_admissions,substring(admittime,1,4) as year
from hosp.d_icd_diagnoses join hosp.diagnoses_icd on hosp.d_icd_diagnoses.icd_code=hosp.diagnoses_icd.icd_code and hosp.d_icd_diagnoses.icd_version=hosp.diagnoses_icd.icd_version join hosp.admissions on hosp.diagnoses_icd.hadm_id=hosp.admissions.hadm_id
where long_title ILIKE '%infection%'
group by hosp.diagnoses_icd.subject_id,year
having count(distinct hosp.admissions.hadm_id)>1
order by year,count_admissions desc,subject_id