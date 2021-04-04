"""
INTELIGENCIA ARTIFICIAL 2021


TAREA
A partir de los ejemplos vistos, buscar un nuevo dataset, evaluar variables y analizar los resultados obtenidos.

El Clasificador Bayesiano , ¿Se adapta de forma óptima a su conjunto de datos? ¿Por qué?


PLANTEAMIENTO DEL EJERCICIO
Se reutilizó el código fuente de ejemplo para un clasificador bayesiano con la misión de determinar el género al que pertenece una canción,
conociendo un fragmento la letra (https://github.com/LuisAlejandroSalcedo/StringTagger-Clasificador-de-Texto).
Se buscaron 5 ejemplos representativos de 3 géneros musicales para intentar encontrar un patrón en las letras, y a partir de ahí, suministrarle al algoritmo
ciertas frases que pertenecen a varias canciones, para intentar determinar el estilo más probable de música al que pertenece.

El dataset en este caso son páginas web cuyo contenido abarca la letra de cada canción, y las etiquetas son los géneros musicales.

LIMITACIONES
De antemano se conocía que, dados los pocos ejemplos que le suministramos al clasificador, podría llegar a haber dificultad en la identificación,
ya que ciertos géneros musicales pueden llegar a tratar asuntos similares en sus letras.
Los resultados muestran que el clasificador se anticipa bastante bien a ciertas combinaciones de frases, pero en otras la diferencia está demasiado difusa
como para dar una respuesta válida en cada caso.

RESULTADOS
El clasificador acertó en el 50% de los casos en el género de la canción.
No se adapta óptimamente al conjunto de datos, tal vez falta un dataset más grande, o bien no hay un verdadero patrón para analizar en los datos.
El tipo de datos que estamos analizando puede llegar a ser más ambiguo de lo que creemos.



TRABAJO INDIVIDUAL: Fernando Vatri
REALIZADO EN VISUAL STUDIO 2019 COMMUNITY
"""

__author__ = "Luis Salcedo" 

from StringTagger.StringClf import Classifier
from StringTagger.getPage import getTextPage

training_data = { # Datos para entrenar al clasificador
	"BALADAS":[
		'https://www.musica.com/letras.asp?letra=4911',
		'https://www.musica.com/letras.asp?letra=103446',
		'https://www.musica.com/letras.asp?letra=2204159',
		'https://www.musica.com/letras.asp?letra=1001586',
		'https://www.musica.com/letras.asp?letra=74834'
	],
	"REGGAETÓN":[
		'https://www.musica.com/letras.asp?letra=2469622',
		'https://www.musica.com/letras.asp?letra=1036495',
		'https://www.musica.com/letras.asp?letra=2364747',
		'https://www.musica.com/letras.asp?letra=2353249',
		'https://www.musica.com/letras.asp?letra=1044531'
	],
	"ROCK NACIONAL":[
		'https://www.musica.com/letras.asp?letra=1300707',
		'https://www.musica.com/letras.asp?letra=842431',
		'https://www.musica.com/letras.asp?letra=71197',
		'https://www.musica.com/letras.asp?letra=117424',
		'https://www.musica.com/letras.asp?letra=59587'
	],
	
}

clf = Classifier() # Instancia del clasificador

for category,urls in training_data.items(): # Entrenamos al clasificador con el contenido de cada pagina
	for url in urls:
		clf.train(getTextPage(url),category) # El metodo "getTextPage", recive como argumento una url para extraer su texto

# Iniciamos el proceso de clasificación con el metodo "String"
# Solo le pasamos como argumento el texto que deseamos etiquetar (clasificar)
string = "En cambio no hoy no hay tiempo de explicarte y preguntar si te amé lo suficiente yo estoy aquí y quiero hablarte ahora"
clas = clf.String(string)
print('\n')
print("Texto: %s " % string)
print("Etiqueta esperada: BALADAS")
print("Etiqueta del Texto: %s" % clas)

string = "En palabras simples y comunes yo te extraño, En lenguaje terrenal mi vida eres tú, En total simplicidad sería yo te amo"
clas = clf.String(string)
print('\n')
print("Texto: %s " % string)
print("Etiqueta esperada: BALADAS")
print("Etiqueta del Texto: %s" % clas)

string = "Un remolino mezcla Los besos y la ausencia Imágenes paganas Se desnudarán en sueños"
clas = clf.String(string)
print('\n')
print("Texto: %s " % string)
print("Etiqueta esperada: ROCK NACIONAL")
print("Etiqueta del Texto: %s" % clas)

string = "Que seas feliz con él yo no te contestaré Sé que me vas a llamar cuando me extrañe tu piel"
clas = clf.String(string)
print('\n')
print("Texto: %s " % string)
print("Etiqueta esperada: REGGAETÓN")
print("Etiqueta del Texto: %s" % clas)

string = "Despacito, quiero respirar tu cuello despacito deja que te diga cosas al oído para que te acuerdes si no estás conmigo"
clas = clf.String(string)
print('\n')
print("Texto: %s " % string)
print("Etiqueta esperada: REGGAETÓN")
print("Etiqueta del Texto: %s" % clas)

string = "Nada cambiara Con un aviso de curva En sus caras veo el temor Ya no hay fabulas En la ciudad de la furia"
clas = clf.String(string)
print('\n')
print("Texto: %s " % string)
print("Etiqueta esperada: ROCK NACIONAL")
print("Etiqueta del Texto: %s" % clas)