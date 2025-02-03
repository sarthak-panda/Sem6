
with recursive
T1 as (
	select subject_id,hadm_id,admittime,dischtime
	from hosp.admissions
	order by admittime
	limit 200
)
,T2 as (
	select distinct x.subject_id as source,y.subject_id as target
	from T1 x, T1 y
	where ((x.admittime::TIMESTAMP > y.admittime::TIMESTAMP and x.admittime::TIMESTAMP < y.dischtime::TIMESTAMP) or (y.admittime::TIMESTAMP > x.admittime::TIMESTAMP and y.admittime::TIMESTAMP < x.dischtime::TIMESTAMP))
)
, shortest_path AS (
    -- Base case: Start from the source node '10037861'
    SELECT source, target, 1 AS path_length, 
        source || '->' || target AS path, 
        ARRAY[source, target] AS visited_nodes -- Track visited nodes
    FROM T2
    WHERE source = '10037861'
    
    UNION ALL

    -- Recursive case: Expand paths without revisiting nodes
    SELECT sp.source, e.target, sp.path_length + 1, 
        sp.path || '->' || e.target, 
        sp.visited_nodes || e.target -- Append new node to visited list
    FROM shortest_path sp
    JOIN T2 e ON sp.target = e.source
    WHERE sp.path_length < 200 -- Prevent infinite loops
    AND NOT (e.target = ANY(sp.visited_nodes)) -- Ensure target isn't visited
)
SELECT source as start_id,target as connected_id,path_length
FROM shortest_path
--WHERE path_length = (select min(path_length) from shortest_path)
ORDER BY path_length,connected_id