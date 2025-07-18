select x.*, 'folder' as rtype,
case when y.id is null then 1
else 0 end as is_leaf
from
(select * from folder
where parentid is null) x left join  
(select unique a.* from 
	folder a left join folder b 
		on a.id = b.parentid 
where b.id is not NULL) y 
on x.id = y.id
