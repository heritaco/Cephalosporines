Tenemos 908033201 reportes desde 1965 hasta 2025.

En donde 17_890 reportes tuvieron efectos adversos y consumieron cefalosporinas. 

En esta notebook solo salvo los archivos limpios de cephalosporinas. Ver la priemera seccion `Drug Product Ingredients` para ver como se hizo el filtrado. Ver la seccion de `Report Drug` para ver como se obtuvieron los reportes relacionados a cephalosporinas.

En `Drug Product Ingredients` se obtuvieron los nombres de medicamentos que usan cephalosporinas (tuvieran *ceph* o *cef* como ingrediente activo) y luego se usaron para filtrar los reportes en `Report Drug`. Finalmente, se usaron los `REPORT_ID` resultantes para filtrar las demas tablas y obtener solo la informacion relacionada a cephalosporinas.