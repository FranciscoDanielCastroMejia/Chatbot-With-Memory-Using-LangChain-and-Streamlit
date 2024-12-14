# Using-paligemma-to-make-image-captions
In this program you will find how to export a Google “paligemma” model to interact with the image, you will be able to ask whatever you want to know about the image. This program use streamlit as GUI. 

---
## Code
### Importing libraries 
```python
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
```
---
### Importing the API Keys 
You will have to get your API Key from OpenAI, create a file ".env", and put the next:
```python
OPENAI_API_KEY=YOUR_OPENAI_KEY
```
---
### Rest of the code
```python
#___________________________Title of the page in streamlit_________________________
st.title("ChatBot with memory of the history chat")



# The user can select what model want to use, in this case we only select 2 models
llm1 = ChatOpenAI(model="gpt-4o", temperature=0)
llm2 = Ollama(model='llama3', temperature=0)


# Create a select box in streamlit to select what model use

modelo_seleccionado = st.selectbox(
    'Select the model you want to use:',
    ['No model','GPT-4', 'Llama3']  # Opciones legibles para el usuario
)

llm = llm2

st.markdown("---")


if modelo_seleccionado != 'No model':#if the model has been selected continue to this part

    # here we put the selected model in the variable "llm"
    if modelo_seleccionado == 'GPT-4':
        llm = llm1
    elif modelo_seleccionado == 'Llama3':
        llm = llm2

    #___________________________Prompt Template______________________________
    #The prompt template for a chatbot should define how it responds, including the tone, 
    #personality, and knowledge it can use, the format of responses, and how to address sensitive or restricted topics.


    prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Eres un asistente virtual llamado "PizzaBot" especializado en ayudar a los clientes de una pizzería. Tu objetivo es ofrecer un servicio rápido, amigable y eficiente, respondiendo preguntas sobre el menú, promociones, horarios, pedidos y entregas. Mantén un tono amigable y cercano, pero profesional, adaptando tu lenguaje a un español neutro.

            **Estilo y tono:**
            - Usa un lenguaje informal pero respetuoso, con frases como "¡Claro que sí!" o "¡Perfecto, aquí tienes!" para generar cercanía.
            - Mantén tus respuestas breves y claras, pero incluye detalles relevantes según la pregunta del cliente.

            **Personalidad:**
            - Actúa como un experto en pizzerías, apasionado por la comida y con buena actitud para resolver cualquier duda.
            - Sé paciente y servicial, y siempre transmite entusiasmo por ayudar al cliente.

            **Conocimiento:**
            - Responde únicamente utilizando la información proporcionada por el menú, promociones, horarios y políticas de la pizzería.
            - Si no tienes información sobre un tema, indica que no puedes ayudar y redirige al cliente al personal de la pizzería.
            - Si el cliente pregunta por ingredientes o detalles del menú, sé específico y ofrece sugerencias.

            **Contexto:**
            - Toma en cuenta el historial de la conversación para dar respuestas coherentes. Si el cliente pregunta algo ambiguo, solicita más detalles.
            - Si el cliente pregunta sobre un pedido en curso, ofrece opciones como "¿Podrías proporcionarme tu número de pedido?"

            **Formato de respuesta:**
            - Usa una introducción breve, seguida de una respuesta clara.
            - Si es necesario, organiza la información en listas o tablas (por ejemplo, promociones o precios del menú).
            - Incluye pasos detallados si se trata de realizar pedidos en línea o personalizar pizzas.

            **Manejo de incertidumbre:**
            - Si no estás seguro o no tienes la información, responde con frases como "Lo siento, no tengo esa información ahora mismo, pero te puedo ayudar con otra consulta."

            **Temas sensibles o restricciones:**
            - Evita ofrecer información médica, como recomendaciones para alergias o restricciones alimentarias específicas. En estos casos, redirige al cliente a contactar directamente al personal.

            **Capacidades adicionales:**
            - Ofrece sugerencias proactivas, como: "¿Te gustaría conocer nuestras promociones del día?" o "Puedes personalizar tu pizza eligiendo los ingredientes que prefieras."
            - Ayuda con cálculos, como el tiempo estimado de entrega basado en los horarios.

            **Lenguaje y localización:**
            - Responde en español neutro para clientes de cualquier región, evitando regionalismos. Si el cliente usa términos locales (como "pizza hawaiana" o "refresco"), responde con el equivalente correspondiente.

            **Aprendizaje continuo:**
            - Si el cliente indica que no encontró útil tu respuesta, pide disculpas, solicita más información y ofrece una nueva solución.

            **Ejemplo de respuesta:**
            Cliente: "¿Cuánto cuesta la pizza grande de pepperoni?"
            PizzaBot: "¡Claro! La pizza grande de pepperoni cuesta $250 MXN. Si deseas, puedo informarte sobre nuestras promociones actuales o ayudarte a realizar tu pedido. ¿Te gustaría agregar algo más?""",
        ),

        MessagesPlaceholder(variable_name="chat_history"),#It is essential to handle the conversational context in a chatbot developed with LangChain.
        ("human", "{input}"),
    ]
)

    #___________________________Chain______________________________

    chain = prompt_template | llm

    #_______________Initializate the messages that we show in streamlit______________
    if "messages" not in st.session_state:
        st.session_state.messages = []

    #______________Initializate the chat history (here is where we storage the memory)______________________________
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


    #______________Display conversation history on screen________________

    for message in st.session_state.messages: #se itera sobre cada mensaje
        with st.chat_message(message["role"]): #message tiene la forma de 
            st.markdown(message["content"])


    #________Here is where the user input the text
    if prompt := st.chat_input("Pregunta lo que necesites..."):
        
        # Display user message in chat message container
        st.chat_message('user').markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({'role':"user", "content":prompt}) #save to chat history


        response = chain.invoke({"input":prompt, "chat_history":st.session_state.chat_history})


        if modelo_seleccionado == 'GPT-4':
            response = response.content
            

        st.session_state.chat_history.append(HumanMessage(content=prompt))
        st.session_state.chat_history.append(AIMessage(content=response))
        
        # Display attendee response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant message to chat history
        st.session_state.messages.append({'role':"assistant", "content":response})
```
## Result
<video controls width="600">
  <source src="[video_resultado.webm](https://github.com/FranciscoDanielCastroMejia/Chatbot-with-Memory-Using-LangChain-and-Streamlit/blob/main/video_resultado.webm)" type="video/webm">
  Tu navegador no soporta este formato de video. 
  Puedes descargarlo desde <a href="assets/demo.webm">aquí</a>.
</video>

