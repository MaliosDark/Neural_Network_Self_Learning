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



from homero import *

# Crear o cargar el objeto RedNeuronal con el chat_id existente o generado automáticamente
red_neuronal = RedNeuronal('')
if red_neuronal.chat_id == '':
    red_neuronal.obtener_chat_id()
    while red_neuronal.chat_id == '':  # Esperar hasta que se obtenga un chat_id
        time.sleep(5)  # Esperar 5 segundos antes de volver a verificar
else:
    # Intentar cargar el cerebro existente o crear uno nuevo si no existe
    red_cargado = RedNeuronal.cargar_cerebro()
    if red_cargado:
        red_neuronal = red_cargado
    else:
        red_neuronal.guardar_cerebro()  # Crear un nuevo cerebro

    # Iniciar proceso de aprendizaje con preguntas iniciales
    red_neuronal.iniciar_aprendizaje()

while not red_neuronal.entrenamiento_completo:
    red_neuronal.auto_extension_codigo()
    red_neuronal.aprender_autonomamente()
    red_neuronal.guardar_cerebro()
    print(f"Estado de entrenamiento: {'Completo' if red_neuronal.entrenamiento_completo else 'En progreso'}")
    time.sleep(20)  # Esperar 20 segundos antes de iniciar el siguiente cicloF