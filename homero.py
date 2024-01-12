"""
*********************************************
*    Personalized Neural Network Script     *
*********************************************

Author: [Andryu Schittone]
Email: [malios666@gmail.com]
GitHub: [https://github.com/MaliosDark/]

Description:
This Python script implements a neural network with autonomous learning capabilities. The code interacts with the Serge Chat API, leveraging the power of language models for code improvement and autonomous learning. The neural network is trained to enhance its capabilities over time and provides valuable insights into code optimization.

Usage:
1. Make sure to set up the Serge Chat API with your IP address and port.
2. Install the required dependencies using the provided requirements.txt file.
3. Run the script using the command: python mainy.py.

Important Files:
- cerebro.pkl: Stores the current state of the neural network.
- codigo_actual.txt: Contains the extracted code from conversations with the language model.

Additional Notes:
- Customize URLs, prompts, or parameters according to your specific use case.
- Explore the autonomous learning and code enhancement features of this neural network.

Feel free to reach out for any questions or collaboration opportunities.

Happy coding!
"""

import os
import random
import re
import threading
import requests
from urllib.parse import quote
import pickle
import time
import autopep8


class RedNeuronal:
    def __init__(self, chat_id):
        self.chat_id = chat_id
        self.entrenamiento_completo = False
        self.retroalimentacion_positiva = 0
        self.retroalimentacion_negativa = 0
        self.codigo_actual = ''
        self.confianza_IA = 0.5  # Valor inicial de confianza

    def obtener_chat_id(self):
        """
        Obtiene el ID del chat al conectarse al servicio de chat.

        Realiza una solicitud HTTP POST al servicio de chat con un prompt predefinido.
        Si la solicitud es exitosa, almacena el ID del chat en el atributo 'chat_id'.
        """
        try:
            pre_prompt = "You are a Python programming language expert. You have extensive knowledge and experience in Python development. You're proficient in various Python libraries such as NumPy, Pandas, and TensorFlow. Your expertise includes data manipulation, machine learning algorithms, and deep learning architectures. You are constantly seeking ways to optimize and enhance code performance. You have a deep understanding of Python syntax, object-oriented programming, and best practices in software development. You are eager to teach and adapt your Python skills to improve this neural network's codebase. Please provide guidance and instructions on code enhancement, best practices, and innovative techniques to elevate the capabilities of this neural network."
            url = f"http://192.168.68.24:8008/api/chat/?model=Mixtral-8X7B-Instruct-v0_1&temperature=0.1&top_k=50&top_p=0.95&max_length=2048&context_window=2048&repeat_last_n=64&repeat_penalty=1.3&n_threads=42&init_prompt={quote(pre_prompt)}&gpu_layers=0"
            #url = f"https://192.168.68.24:8008/api/chat/?model=Zephyr-7B-Beta&temperature=0.1&top_k=50&top_p=0.95&max_length=2048&context_window=2048&repeat_last_n=64&repeat_penalty=1.3&n_threads=28&init_prompt={quote(pre_prompt)}&gpu_layers=0"

            response = requests.post(url, headers={'Accept': 'application/json'})
            if response.ok:
                self.chat_id = response.json()
                print(f"Connected to chat with ID: {self.chat_id}")
            else:
                print('Request error:', response.text)
        except Exception as e:
            print('Error creating chat:', e)

    def enviar_pregunta_al_modelo(self, pregunta):
        """
        Envía una pregunta al modelo de lenguaje y devuelve la respuesta limpia.

        Realiza una solicitud HTTP GET al servicio de chat con la pregunta proporcionada.
        Limpia la respuesta antes de retornarla utilizando la función 'clean_response'.
        """
        chat_url = f"http://192.168.68.24:8008/api/chat/{self.chat_id}/question?prompt={quote(pregunta)}"
        try:
            response = requests.get(chat_url, headers={'Accept': 'text/plain'})
            if response.ok:
                return self.clean_response(response.text)  # Limpia la respuesta antes de retornarla
            else:
                return "No se pudo obtener una respuesta del modelo de lenguaje."
        except Exception as e:
            return f"Error en la solicitud al modelo de lenguaje: {e}"

    def clean_response(self, data, estilo='autopep8'):
        """
        Limpia la respuesta del modelo de lenguaje.

        Realiza varios pasos de limpieza en la respuesta obtenida del modelo.
        Puede aplicar el formateo según el estilo especificado (por defecto, autopep8).
        """
        try:
            clean_data = re.sub(r'data:|event: message|event: close|: ping - \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}.\d{6}', '', data)
            trimmed_data = ' '.join(clean_data.split())
            remove_instructions = re.sub(r'##\s?#\s?Instruction\s?:[^#]+|##\s?#\s?Response\s?:[^#]+|##\s?#\s?[^#]+:\s?[^#]+', '', trimmed_data)
            clean_content = remove_instructions.strip()

            if estilo == 'autopep8':
                clean_content = autopep8.fix_code(clean_content)
                print("Código formateado según PEP 8.")

            return clean_content

        except Exception as e:
            print(f"Error al limpiar la respuesta: {e}")
            return clean_content
        
    ###########################
    def iniciar_aprendizaje(self):
        # Start the learning process by asking initial questions
        preguntas_iniciales = [
            "What information can you provide about your current environment?",
            "What is your main function?",
            "What are your long-term goals?",
            "How would you like to improve or continue learning?"
        ]
        for pregunta in preguntas_iniciales:
            # Ask questions and get responses
            respuesta = self.enviar_pregunta_al_modelo(pregunta)
            # Display the question and response
            print(f"Pregunta: {pregunta}\nRespuesta: {respuesta}\n")
            # Process the response for learning (add your logic here)

    def procesar_respuesta_aprendizaje(self, respuesta):
        # Process the response received during the learning process
        if "positiva" in respuesta:
            self.retroalimentacion_positiva += 1
        elif "negativa" in respuesta:
            self.retroalimentacion_negativa += 1

        # Extract relevant data from the response
        datos_relevantes = self.extraer_datos(respuesta)
        if datos_relevantes:
            self.usar_datos(datos_relevantes)

    def extraer_datos(self, respuesta):
        # Extract relevant data patterns from the response
        datos = None

        # Search for relevant patterns in the response
        patron_codigo = r"(?<!\w)(?:class|def|for|while|if|else|import)(?=\s)"
        patron_sugerencias = r"(?:suggestions?|improvements?|fixes?)(?=\W)"
        patron_datos = r"(?:data|information|details|code)(?=\W)"

        if re.search(patron_codigo, respuesta, re.IGNORECASE):
            datos = "code"

        if re.search(patron_sugerencias, respuesta, re.IGNORECASE):
            datos = "improvements"

        if re.search(patron_datos, respuesta, re.IGNORECASE):
            datos = "data"

        return datos

    def usar_datos(self, datos):
        # Use the extracted data to make adjustments
        if datos == "aumentar_confianza":
            self.aumentar_confianza_IA()
        elif datos == "reducir_confianza":
            self.reducir_confianza_IA()
        else:
            pass  # Additional logic based on the obtained data

    def aumentar_confianza_IA(self):
        # Increase confidence in the intelligent assistant
        factor_ajuste = 0.1
        self.confianza_IA += factor_ajuste

    def reducir_confianza_IA(self):
        # Decrease confidence in the intelligent assistant
        factor_ajuste = 0.1
        self.confianza_IA -= factor_ajuste

    def ajustar_IA(self):
        # Adjust the intelligent assistant based on feedback
        if self.retroalimentacion_positiva > self.retroalimentacion_negativa:
            self.usar_datos("aumentar_confianza")
        elif self.retroalimentacion_positiva < self.retroalimentacion_negativa:
            self.usar_datos("reducir_confianza")
        else:
            pass  # No significant changes needed

    def obtener_confianza_IA(self):
        # Get the initial confidence level
        return 0.5  # Initial confidence value

    def actualizar_confianza_IA(self, nueva_confianza):
        # Update the confidence level, ensuring it is within a valid range (e.g., between 0 and 1)
        if 0 <= nueva_confianza <= 1:
            self.confianza_IA = nueva_confianza
            print(f"Nivel de confianza actualizado a: {self.confianza_IA}")
        else:
            print("Error: El valor de confianza debe estar entre 0 y 1.")

    #################

    #################
    def auto_extension_codigo(self):
        # Método principal para mejorar el código
        pregunta_codigo = "Can you provide more code to improve?"
        respuesta_codigo = self.enviar_pregunta_al_modelo(pregunta_codigo)
        print(f"Pregunta Código: {pregunta_codigo}\nRespuesta Código: {respuesta_codigo}\n")

        # Procesar la respuesta y mejorar el código actual
        self.procesar_respuesta_codigo(respuesta_codigo)

    def procesar_respuesta_codigo(self, respuesta):
        # Procesar la respuesta recibida
        if any(keyword in respuesta for keyword in ["code", "suggestions"]):
            # Aplicar mejoras al código existente
            codigo_mejorado = self.aplicar_mejoras(respuesta)

            # Escanear el código actual y buscar áreas para mejorar
            areas_por_mejorar = self.escanear_codigo(codigo_mejorado)

            # Ajustar el formato del código si es necesario
            codigo_formateado = self.ajustar_formato(codigo_mejorado)

            # Actualizar el código actual con las mejoras
            self.actualizar_codigo(codigo_formateado)

            # Utilizar areas_por_mejorar para registro o procesamiento adicional
            print("Áreas identificadas para mejorar:", areas_por_mejorar)
        else:
            # La respuesta no contiene código o sugerencias para mejorar
            pass

    def escanear_codigo(self, codigo_actual):
        # Escanear el código para identificar áreas a mejorar
        areas_por_mejorar = []

        # Escaneo para optimizar bucles (loops)
        areas_por_mejorar.extend(self.analizar_optimizacion_bucles(codigo_actual))

        # Escaneo para corregir nombres de variables
        areas_por_mejorar.extend(self.analizar_nombres_variables(codigo_actual))

        # Escaneo para agregar comentarios
        areas_por_mejorar.extend(self.analizar_comentarios(codigo_actual))

        # Otros análisis y escaneos pueden agregarse según sea necesario

        return areas_por_mejorar

    def analizar_optimizacion_bucles(self, codigo):
        # Analizar y devolver áreas para optimizar bucles
        areas_optimizar_bucles = []
        for linea in codigo.split('\n'):
            if 'for' in linea and 'range' in linea:
                areas_optimizar_bucles.append(linea)
        return areas_optimizar_bucles

    def analizar_nombres_variables(self, codigo):
        # Analizar y devolver áreas para corregir nombres de variables
        areas_corregir_nombres = []
        for linea in codigo.split('\n'):
            palabras = linea.split()
            for palabra in palabras:
                if len(palabra) <= 3:
                    areas_corregir_nombres.append(palabra)
        return areas_corregir_nombres

    def analizar_comentarios(self, codigo):
        # Analizar y devolver áreas para agregar comentarios
        areas_agregar_comentarios = []
        lineas = codigo.split('\n')
        for i in range(len(lineas)):
            if lineas[i].strip() != '' and i < len(lineas) - 1 and lineas[i + 1].strip() == '':
                areas_agregar_comentarios.append(lineas[i])
        return areas_agregar_comentarios

    def aplicar_mejoras(self, respuesta):
        # Aplicar las mejoras sugeridas por la respuesta
        codigo_mejorado = self.codigo_actual + respuesta
        return codigo_mejorado

    def ajustar_formato(self, codigo, estilo='autopep8'):
        # Ajustar el formato del código
        try:
            if estilo == 'autopep8':
                # Aplicar formato PEP 8 utilizando autopep8
                codigo_formateado = autopep8.fix_code(codigo)
                print("Código formateado según PEP 8.")
                return codigo_formateado
            elif estilo == 'indentacion':
                # Aplicar únicamente indentación
                codigo_formateado = "\n".join(line.rstrip() for line in codigo.split("\n"))
                print("Indentación aplicada al código.")
                return codigo_formateado
            else:
                print("Estilo de formato no reconocido.")
                return codigo
        except Exception as e:
            print(f"Error al ajustar el formato del código: {e}")
            return codigo

    def actualizar_codigo(self, codigo):
        # Actualizar el código actual con el código mejorado
        self.codigo_actual = codigo
        # Guardar el código actualizado o realizar otros pasos necesarios
        self.guardar_codigo_actual()

    def guardar_codigo_actual(self):
        # Guardar el código actualizado en un archivo
        try:
            nombre_archivo = 'codigo_actual.txt'
            if os.path.exists(nombre_archivo):
                modo_apertura = 'a'
            else:
                modo_apertura = 'w'
            with open(nombre_archivo, modo_apertura) as file:
                file.write(self.codigo_actual)
                print("Código actualizado guardado exitosamente en el archivo.")
        except Exception as e:
            print(f"Error al guardar el código actualizado: {e}")


    #########
            
    #########

    def generar_pregunta_autonoma(self):
        # Verificar si el entrenamiento está completo
        if self.entrenamiento_completo:
            # Lista de preguntas automáticas
            preguntas_automaticas = [
                "What could be a potential optimization strategy for the current problem?",
                "How might this solution generalize to other similar scenarios?",
                "What alternative approaches could be explored?",
                "Are there any recent advancements in the field that could be applied here?",
                "What potential drawbacks might exist in the current implementation?",
                "How does this solution align with industry best practices?",
                # Agregar más preguntas relevantes basadas en el contexto del entrenamiento
            ]

            # Seleccionar una pregunta al azar
            pregunta_generada = random.choice(preguntas_automaticas)
            respuesta_generada = self.enviar_pregunta_al_modelo(pregunta_generada)

            # Procesar la respuesta generada si es necesario
            self.procesar_respuesta_autonoma(respuesta_generada)

            # Devolver la pregunta y respuesta generadas
            return pregunta_generada, respuesta_generada
        else:
            # Mensaje si el entrenamiento no está completo
            return "El entrenamiento aún no está completo. Por favor, espere hasta que se complete el proceso."

    def aprender_autonomamente(self):
        # Verificar si el entrenamiento no está completo
        if not self.entrenamiento_completo:
            # Lista de preguntas automáticas
            preguntas_automaticas = [
                "Could you elaborate more on the positive aspects?",
                "Could you explain more about the negative aspects?",
                "What other insights can you provide?",
                "What specific challenges did you encounter?",
                "How does this relate to your previous experiences?",
                "In what ways do you think this could be improved?",
                "Can you provide more details on that particular point?",
                "What impact do you foresee from these actions?",
                "Are there any alternative approaches you considered?",
                "How do you think this might affect future outcomes?"
            ]

            # Seleccionar una pregunta al azar
            pregunta = random.choice(preguntas_automaticas)

            # Enviar la pregunta al modelo y obtener la respuesta
            respuesta = self.enviar_pregunta_al_modelo(pregunta)
            
            # Imprimir la pregunta y respuesta
            print(f"Pregunta Autónoma: {pregunta}\nRespuesta Autónoma: {respuesta}\n")

            # Procesar la respuesta para el aprendizaje (agrega tu lógica aquí)
            self.procesar_respuesta_aprendizaje(respuesta)

        else:
            # Mensaje si el entrenamiento está completo
            print("El entrenamiento ya está completo.")

    def procesar_respuesta_aprendizaje(self, respuesta):
        # Llamar a la función analizar_respuesta
        self.analizar_respuesta(respuesta)

    def procesar_respuesta_autonoma(self, respuesta):
        # Llamar a la función analizar_respuesta
        self.analizar_respuesta(respuesta)

    def analizar_respuesta(self, respuesta):
        # Verificar si la respuesta es positiva o negativa y actualizar contadores
        if "positiva" in respuesta:
            self.retroalimentacion_positiva += 1
        elif "negativa" in respuesta:
            self.retroalimentacion_negativa += 1

        # Extraer y usar datos relevantes
        datos_relevantes = self.extraer_datos(respuesta)
        if datos_relevantes:
            self.usar_datos(datos_relevantes)

    #######################
    

    #######################

    def ciclo_entrenamiento(self):
        epoch = 0  # Contador de épocas
        while not self.entrenamiento_completo:
            self.auto_extension_codigo()
            self.aprender_autonomamente()
            self.guardar_cerebro()

            # Actualizar estado de entrenamiento
            epoch += 1
            tiempo_transcurrido = epoch * 20  # Cada ciclo es de 20 segundos
            calidad_entrenamiento = self.calcular_calidad()  # Función que calcula la calidad del entrenamiento
            print(f"Epoch: {epoch}, Tiempo transcurrido: {tiempo_transcurrido} segundos, Calidad: {calidad_entrenamiento}")

            # Añade la llamada a la función generar_pregunta_autonoma()
            pregunta_generada, respuesta_generada = self.generar_pregunta_autonoma()
            print(f"Pregunta Autónoma: {pregunta_generada}\nRespuesta Autónoma: {respuesta_generada}\n")
            
            print(f"Estado de entrenamiento: {'Completo' if self.entrenamiento_completo else 'En progreso'}")
            time.sleep(20)  # Esperar 20 segundos antes de iniciar el siguiente ciclo


    def calcular_calidad(self):
        #  métricas reales para evaluar el entrenamiento

        # Ejemplo de métricas comunes: precisión, pérdida, puntaje F1
        precision = self.calcular_precision()  # Función que calcula la precisión
        perdida = self.calcular_perdida()  # Función que calcula la pérdida
        puntaje_f1 = self.calcular_puntaje_f1()  # Función que calcula el puntaje F1

        # Podrías ponderar estas métricas o utilizar cualquier otro método para calcular la calidad
        calidad = (precision + puntaje_f1) / 2 - perdida  # Fórmula de calidad de ejemplo

        return calidad

    def calcular_precision(self):
        # Lógica para calcular la precisión
        # Puede ser mediante evaluación con datos de prueba o validación cruzada
        # Ejemplo: métrica de precisión del modelo
        precision = 0.85  # Ejemplo de precisión
        return precision

    def calcular_perdida(self):
        # Lógica para calcular la pérdida
        # Puede ser la función de pérdida utilizada durante el entrenamiento
        # Ejemplo: pérdida del modelo
        perdida = 0.3  # Ejemplo de pérdida
        return perdida

    def calcular_puntaje_f1(self):
        # Lógica para calcular el puntaje F1
        # Puede ser otra métrica de evaluación del modelo
        # Ejemplo: puntaje F1 del modelo
        puntaje_f1 = 0.78  # Ejemplo de puntaje F1
        return puntaje_f1

    def guardar_cada_5_minutos(self):
        threading.Timer(300, self.guardar_cada_5_minutos).start()  # 300 segundos = 5 minutos
        self.guardar_cerebro()

    def guardar_cerebro(self):
        try:
            with open('cerebro.pkl', 'wb') as file:
                pickle.dump(self, file)
                print("Cerebro guardado exitosamente.")
        except Exception as e:
            print(f"Error al guardar el cerebro: {e}")


    @staticmethod
    def cargar_cerebro():
        try:
            with open('cerebro.pkl', 'rb') as file:
                red_neuronal = pickle.load(file)
                print("Cerebro cargado exitosamente.")
                return red_neuronal
        except FileNotFoundError:
            print("No se encontró ningún cerebro existente.")
            return None
        except Exception as e:
            print(f"Error al cargar el cerebro: {e}")
            return None
