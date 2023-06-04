import sys
from langchain.memory import ConversationBufferMemory
from langchain import HuggingFacePipeline, PromptTemplate, LLMChain

# Read template from file
with open("./templates/chat.tpl") as f:
    template = f.read()

# Build prompt by template
prompt = PromptTemplate(template=template, input_variables=["chat_history", "human_input"])

# Init memory of chat
memory = ConversationBufferMemory(
    memory_key="chat_history",
    human_prefix="Рогожин",
    ai_prefix="Князь",
    # return_messages=True
)

# Init model from folder and pass some parameters
rugpt = HuggingFacePipeline.from_model_id(
    # The name of the model on the HuggingFace server (for example "gpt2" by default) or the path to the model on disk
    model_id="./models/dostoevsky_doesnt_write_it",
    # Other options are possible: "text2text-generation", "text-generation" (default), "summarization"
    task="text-generation",
    # Settings available for the model
    pipeline_kwargs={
        "temperature": 1.0,
        "max_length": 2048,
        "repetition_penalty": 1.2,
        "do_sample": True,
        "top_k": 5,
        "top_p": 0.95,
    },
    # Use GPU
    device=0,
)

# Run chain for model
rugpt_chain = LLMChain(
    prompt=prompt,
    llm=rugpt,
    verbose=True,  # Display more details about prompt
)

# Run prediction in chat format
question = ""
while question != "exit":

    # Read user text in loop
    question = input("Рогожин: ")
    if question == '':
        continue

    # Save request to memory
    chat_history = memory.load_memory_variables({})['chat_history']

    # Predict network response
    prediction = rugpt_chain.predict(
        human_input=question,
        chat_history=chat_history,
    )

    # Parse response
    p_split = prediction.split('  ')
    response = p_split[0].strip()

    # Save response to memory
    memory.chat_memory.add_user_message(question)
    memory.chat_memory.add_ai_message(response)

    # Print response to user
    print("Князь: " + response)
