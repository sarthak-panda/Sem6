
with T1 as (
	select subject_id,hadm_id
	from hosp.admissions
	order by subject_id,admittime
)
, T2 as (
	select subject_id,array_agg(hadm_id) as admissions
	from T1
	group by subject_id
), T3 as (
	select subject_id,hadm_id
	from hosp.diagnoses_icd
	where icd_code LIKE 'E10%' or icd_code LIKE 'E11%'
	group by subject_id,hadm_id
), T4 as (
	select subject_id, admissions[i] as current_adm, admissions[i + 1] as next_adm
	from T2, generate_subscripts(admissions, 1) as i
), T5 as (
select T3.subject_id,T3.hadm_id,T4.next_adm
from T3 join T4 on T3.subject_id=T4.subject_id and T3.hadm_id=T4.current_adm
), T6 as (
	select subject_id,array_agg(distinct hadm_id) as admissions
	from hosp.diagnoses_icd
	where icd_code LIKE 'N18%'
	group by subject_id
), T7 as (
	select T5.subject_id--,T5.hadm_id--,T5.next_adm,T6.admissions as N18doses
	from T5 join T6 on T5.subject_id=T6.subject_id
	where T5.hadm_id = ANY(T6.admissions) or (T5.next_adm is not null and T5.next_adm = ANY(T6.admissions))
), T8 as (
	select distinct subject_id,hadm_id,'diagnoses' as diagnoses_or_procedure,icd_code
	from hosp.diagnoses_icd
	where subject_id in (select subject_id from T7)
	UNION
	select distinct subject_id,hadm_id,'procedures' as diagnoses_or_procedure,icd_code
	from hosp.procedures_icd
	where subject_id in (select subject_id from T7)
)
select subject_id,hadm_id as admission_id,diagnoses_or_procedure,icd_code
from T8
order by subject_id,hadm_id,icd_code,diagnoses_or_procedure