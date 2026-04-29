# Difusor
Probando hacer un difusor

# INSTALACIÓN DE LIBRERÍAS

Para instalar las librerías sigue estos pasos:
1. Crear un entorno virtual o venv con el comando:
    python -m venv venv
2. Activar el venv con:
    venv\Scripts\activate
3. Instalar las librerías en requirements.txt con:
    pip install -r requirements.txt

Nota: 
Todo esto debe hacerse desde la terminal de tu IDE. En VS Code la terminal se crea con: ctrl + shift + ñ.


# EJECUTAR EL PROGRAMA

La ejecución del programa consiste en poner dentro de la terminal el comando:
    python app.py

La integración entre frontend y backend se hizo con FAST API.

Nota: El entrenamiento se encuentra en google colab debido a que mi PC no contaba con GPU para procesar el entrenamiento (Podría tardar hasta días en hacerlo mientras que en colab se hizo en 2 horas aprox).

El enlace al entrenamiento es: https://colab.research.google.com/drive/1S1yTJs3Eap0maSY8lv08sLBzjeOTva0m?usp=sharing

el archivo de ejecución también está anexado al proyecto con el nombre Difusores y la extensión .ipynb

# ARCHIVOS PRESENTES EN EL REPOSITORIO Y SU USO

El repositorio tiene 2 carpetas que contienen los pesos del modelo LoRA y el template de HTML (El front).
También está presente el archivo app.py que tiene el procesamiento del prompt y la generación de la imagen a través de Stable Diffusion, y LoRA con FAST API.

En el archivo .gitignore se excluye la carpeta venv que contiene las librerías y demás aspectos que ayudan al funcionamiento del programa, pero que son muy pesadas para que se carguen en GitHub.

El archivo requiremnts.txt están todas las librerías necesarias para que el programa funcione y sus respectivas versiones de compatibilidad. Cabe destacar que muchas de ellas funcionan para una versión de python superior a las 3.10.

Los archivos train.py y test.py se pueden usar para entrenar el modelo pero, dado que en el pc que tengo disponible no hay GPU, era más factible hacerlo por colab del cual también se encuentra el archivo presente en el repositorio.

Nota:
El archivo de colab guarda los pesos generados en el entrenamiento con LoRA en google Drive y fueron exportados manualmente, no está automatizado para hacerlo por su propia cuenta. Además el archivo app.py no está adaptado al modelo generado por el entrenamiento desde VS Code o ningún IDE diferente, por tanto, si quieres reentrenar el modelo, sugiero que lo hagas desde google colab y descargues la carpeta de LoRA desde tu Google Drive y la actualices en app.py de ser necesario. O que adaptes el código para que se integre con el entrenamiento y testeo de estos archivos (train y test).

Por último tenemos el archivo Difusores.ipynb, este archivo tiene el entrenamiento del modelo LoRA ejecutable en google Colab. Para ejecutarlo sigue estos pasos:
1. Abre colab y da click en subir notebook
2. Selecciona el archivo Disufores.ipynb
3. Ejecutalo
