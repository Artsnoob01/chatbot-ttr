from gpt_index import SimpleDirectoryReader, GPTListIndex, readers, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain import OpenAI
import sys
import os


def construct_index(directory_path):
    # set maximum input size
    max_input_size = 4096
    # set number of output tokens
    num_outputs = 256
    # set maximum chunk overlap
    max_chunk_overlap = 20
    # set chunk size limit
    chunk_size_limit = 600

    print("Constructing index")

    # define LLM
    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-003", max_tokens=num_outputs))
    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

    documents = SimpleDirectoryReader(directory_path).load_data()
    print(len(documents))

    index = GPTSimpleVectorIndex.from_documents(documents)
    #
    # index = GPTSimpleVectorIndex(
    #     documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper
    # )

    print("Index done")
    index.save_to_disk('index.json')

    return index


def ask_lenny():
    index = GPTSimpleVectorIndex.load_from_disk('index.json')
    while True:
        query = input("What do you want to ask Lenny? ")
        response = index.query(query, response_mode="compact")
        print(response.response)
        # display(Markdown(f"Lenny Bot says: <b>{response.response}</b>"))

os.environ["OPENAI_API_KEY"] = "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"

# construct_index('content/')
ask_lenny()