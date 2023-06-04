import sys
from langchain import HuggingFacePipeline, PromptTemplate, LLMChain

# Init model from folder and pass some parameters
rugpt = HuggingFacePipeline.from_model_id(
    # The name of the model on the HuggingFace server (for example "gpt2" by default) or the path to the model on disk
    model_id="./models/dostoevsky_doesnt_write_it",
    # Other options are possible: "text2text-generation", "text-generation" (default), "summarization"
    task="text-generation",
    # Settings available for the model
    pipeline_kwargs={
        "temperature": 1.0,
        "max_length": 500,
        "repetition_penalty": 1.2,
        "do_sample": True,
        "top_k": 5,
        "top_p": 0.95,
    },
)

# Read template from file
with open("./templates/context.tpl") as f:
    template = f.read()

# Build prompt by template
prompt = PromptTemplate(template=template, input_variables=["question"])

# Run chain for model
rugpt_chain = LLMChain(prompt=prompt, llm=rugpt)

# Read question from first argument of script
question = sys.argv[1]

# Run chain
prediction = question + rugpt_chain.run(question)

# Return response
print(prediction)
