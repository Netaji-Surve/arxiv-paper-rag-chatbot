from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

class Response(BaseModel):
    answer: str = Field(description="answer to the question")
    explanation: str = Field(description="explanation about the answer")

llm = ChatOpenAI(model="meta-llama-3-8b-instruct",  # model name inside LM Studio  #Chatbots Agents Q&A Summarization Coding help
                base_url="http://192.168.0.155:1234/v1/",
                api_key="not-needed",  # can be anything
                temperature=0.7)


#llm_with_structure_output = llm.with_structured_output(Response)

def get_llm():
    return llm