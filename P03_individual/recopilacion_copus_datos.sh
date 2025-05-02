#!/bin/bash
# Recopilación de Corpus de Datos de páginas webs
# ------------------------------------------------

# Abrir una terminal y crear un subdirectorio donde se almacenará
# el corpus de datos a recoger. Moverse a él:
mkdir corpus
cd corpus

# Ejecutar el firefox y conectarse a la página web de búsquedas avanzadas en
# Google. Previamente instalar el Add-on
# [Copy Selected Links](https://addons.mozilla.org/es/firefox/addon/copy-selected-links/):

firefox "http://www.google.com/advanced_search?hl=es" & 

# Realizar búsqueda avanzada
# * Introducir en "cualquiera de estas palabras:" las palabras exactas que
# desea que aparezcan en las páginas del corpus a estudiar (por ejemplo:
# "python snake film", pero usar otras diferentes)
# * Seleccionar en "idioma:" seleccionar "inglés"
# * Pinchar en "Busqueda avanzada" para realizar la búsqueda
# * Mediante "Configuración (engranaje) -> Mas ajustes -> Otros ajustes" desactivar "Desplazamiento continuo" 

# Almacenar los enlaces de la página de resultados:
# * Prevaimente intale en Firefox el Add-on [Copy Selected Links](https://addons.mozilla.org/es/firefox/addon/copy-selected-links/)
# * Abrir en un editor cualquiera el fichero "busqueda.lnk", por ejemplo:
gedit busqueda.lnk &
# * En firefox, ir al final de la página de resultados de búsqueda y pinchar en "Mas resultados" las veces necesarias para obtener una página con unos 300 resultados
# * En firefox, sobre la página de resultados, pinchar botón derecho y
#   "Seleccionar todo". Luego nuevamente pinchar botón derecho y
#   "Copy selected links" (habiendo antes instaldo dicho Add-on)
# * Pege el resultado en el editor y guárdelo con el nombre de fichero
#   "busqueda.lnk"

# Filtrar los enlaces de la página de resultados:
# * Mantener solo los enlaces a páginas html que no sean de google ni youtube:
grep -v google busqueda.lnk | grep -v youtube | grep "^http" > descargas.lnk
# * Seleccionar/modificar con un editor si lo considera necesario las urls
#   a descargar:
gedit descargas.lnk &
# * El número de enlaces obtenidos es:
wc -l descargas.lnk

# Descargar las páginas de los enlaces:
# * Crear subdirectorio "html" donde se descargarán las páginas:
mkdir html
# * Descargar uno a uno los enlaces de "descargas.lnk" mediante el comando
#   wget ejecutando en consola el siguiente script bash (puede tardar bastante):
rm -r html/*
while read url
do
  FILE=./html/`echo $url | sed -e "s/\//_/g"`.html
  echo "Descargando $url"
  wget -A htm,html,txt,shtml -a descargas.log -O $FILE "$url"
done < descargas.lnk
# * Si la descarga de algún enlace se atasca eliminar el enlace de
#   "descargas.lnk" y reiniciar el script anterior
# * Eliminar páginas webs que hayan quedado vacías (si hubiera):
find html/ -size 0 -exec rm {} \;
# * Los ficheros descargados son:
ls -l html
# * El número de páginas descargadas es: 
ls html | wc -l