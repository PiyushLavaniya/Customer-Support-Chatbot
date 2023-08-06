import streamlit as st

from langchain.document_loaders import CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from dotenv import load_dotenv


load_dotenv()

#Vectorising the sales response Data

loader = CSVLoader(file_path = 'Customer-Support.csv')
documents = loader.load()

embeddings = OpenAIEmbeddings()

vectorstore = FAISS.from_documents(documents, embeddings)


#now we will make a function for similarity search

def similarity_search(query):
    
    similar_response = vectorstore.similarity_search(query, k = 3)
    
    page_content = [doc.page_content for doc in similar_response]     #we will extract only the page content or main information cause it will return a lot of thigns such as metadata, etc.
    
    #print(page_content)
    
    return page_content


#query = """ I did not receive my order. What to do now? """


#Setting up our LLM and the prompts

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

template = """

You are a world class business development representative. 
I will share a prospect's message with you and you will give me the best answer that 
I should send to this prospect based on past best practies, 
and you will follow ALL of the rules below:

1/ Response should be very similar or even identical to the past best practies, 
in terms of length, ton of voice, logical arguments and other details

2/ If the best practice are irrelevant, then try to mimic the style of the best practice to prospect's message

Below is a message I received from the prospect:
{message}

Here is a list of best practies of how we normally respond to prospect in similar scenarios:
{best_practice}

Please write the best response that I should send to this prospect:

"""

prompt_template = PromptTemplate(
    input_variables = ['message', 'best_practice'],
    template = template
)


chain = LLMChain(llm = llm,
                 prompt = prompt_template)




#getting the response 
def generate_response(message):
    best_practice = similarity_search(message)
    response = chain.run(message = message, best_practice = best_practice)
    
    return response


#response = """ Hi, I could not receive my order. can you help me? """


#results = generate_response(response)

#print(results)


#creating a streamlit app

def main():
    
    st.set_page_config("Customer Support AI", page_icon  = ":Robot:")
    
    st.title("Customer Response Generator AI")
    
    messages = st.text_input("Customer Message")
    
    if messages:
        
        st.write("Generating Response.....")
        response = generate_response(messages)
        
        st.info(response)
        


if __name__ == '__main__':
    main()
    




