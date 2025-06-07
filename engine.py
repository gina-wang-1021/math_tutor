import openai
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, AIMessage

def get_student_level(level):
    # TODO: convert student level to compatible format: beginner, intermediate, advanced
    return level

def get_relevant_chunks(topic, year, query):
    db = Chroma(persist_directory=f"indexes/{topic}", embedding_function=OpenAIEmbeddings())
    retriever = db.as_retriever(search_kwargs={"k": 4, "filter": {"level": year}})
    return retriever.invoke(query)

def call_openai(messages, model="gpt-3.5-turbo"):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        stream=True,
    )
    return response["choices"][0].message.content

def pipeline(year, topic, user_question, history):
    llm_rephrase = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.0)
    rephrase_prompt = PromptTemplate.from_template("""
    Given a chat history and the latest user question which might reference the chat history, 
    reformulate the latest question as a standalone question. 
    Do NOT answer the question. If the question is already standalone, return it as-is.

    Chat history:
    {history}

    Follow-up question:
    {user_question}
    """)

    llm_chain = LLMChain(llm=llm_rephrase, prompt=rephrase_prompt)
    standalone_question = llm_chain.run({
        "user_question": user_question,
        "history": history
    })
    print("Rephrased Question →", standalone_question)
    input("continue?")

    docs = get_relevant_chunks(topic, year, standalone_question)
    print(docs)
    chunks = "\n".join([doc.page_content for doc in docs[:3]])

    llm_answer = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.5)

    answer_prompt = PromptTemplate.from_template("""
    You are an expert tutor. Use the following information to answer the question.
    Be concise and clear. If the answer is not in the context, say so.

    Context:
    {chunks}

    Question:
    {question}
    """)

    answer_chain = LLMChain(llm=llm_answer, prompt=answer_prompt)
    print("grabbing answer")

    response = answer_chain.run({
        "chunks": chunks,
        "question": standalone_question
    })

    print("Final Answer →", response)
    return response

if __name__ == "__main__":
    pipeline("beginner", "basics", "what is 10+10", "")