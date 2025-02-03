
with temp as (
   select subject_id,hadm_id,chartdate
   from hosp.procedures_icd
   WHERE CAST(icd_code AS TEXT) LIKE '0%' 
      OR CAST(icd_code AS TEXT) LIKE '1%' 
      OR CAST(icd_code AS TEXT) LIKE '2%'
   group by subject_id,hadm_id,chartdate
   order by subject_id,hadm_id,chartdate
)
, temp1 as (--wrong and unused first procedure for given icd code
   select subject_id,hadm_id,array_agg(chartdate) AS procedure_start_dates
   from temp
   group by subject_id,hadm_id
), temp1_new as (
   with internal as (
      select subject_id,hadm_id,chartdate
      from hosp.procedures_icd
      group by subject_id,hadm_id,chartdate
      order by subject_id,hadm_id,chartdate
   )
   select subject_id,hadm_id,substring(array_agg(chartdate)::text,2,10) AS first_procedure_start_date
   from internal
   group by subject_id,hadm_id
)
, temp2 as (
   select subject_id,hadm_id,array_agg(distinct substring(starttime,1,10)) as starttimes
   from hosp.prescriptions--or should i use hosp.pharmacy
   group by subject_id,hadm_id
), temp3 as (
   select subject_id
   from hosp.prescriptions
   group by subject_id--As they asked in entire stay so not including hadm_id
   having count(distinct drug)>=2
), temp4 as (
   select distinct on (hadm_id) hadm_id,starttime as last_medication_time
   from hosp.prescriptions
   order by hadm_id,starttime desc   
), temp5 as (
   select temp1.subject_id,temp1.hadm_id,temp1_new.first_procedure_start_date as first_procedure_time--,substring(procedure_start_dates::text,2,10) as first_procedure_time
   from temp1 join temp2 on temp1.subject_id=temp2.subject_id and temp1.hadm_id=temp2.hadm_id join temp1_new on temp1.subject_id=temp1_new.subject_id and temp1.hadm_id=temp1_new.hadm_id
   where temp1.subject_id in (select subject_id from temp3)
   and 
   exists(
      select 1
      from unnest(starttimes) as s
      where s::DATE in (
         select unnest(procedure_start_dates)::DATE
         union
         select unnest(procedure_start_dates)::DATE + INTERVAL '1 day'
      )
   )
), temp6 as (
   select hadm_id,count(distinct icd_code) as distinct_icd_codes_diag
   from hosp.diagnoses_icd
   group by hadm_id
), temp7 as (
   select hadm_id,count(distinct icd_code) as distinct_icd_codes_proc
   from hosp.procedures_icd
   group by hadm_id
)
select subject_id,temp5.hadm_id,COALESCE(distinct_icd_codes_diag,0) as distinct_diagnoses,COALESCE(distinct_icd_codes_proc,0) as distinct_procedures,TO_CHAR(AGE(last_medication_time::TIMESTAMP,first_procedure_time::TIMESTAMP), 'YYYY-MM-DD HH24:MI:SS') as time_gap
from temp5 join temp4 on temp5.hadm_id=temp4.hadm_id left outer join temp6 on temp5.hadm_id=temp6.hadm_id left outer join temp7 on temp5.hadm_id=temp7.hadm_id
order by distinct_diagnoses desc,distinct_procedures desc,time_gap,subject_id,temp5.hadm_id