from pydantic import BaseModel
import os, sys 
import traceback 
import pymongo
from fastapi import (FastAPI, 
                     HTTPException,
                     status, 
                     UploadFile) 
from fastapi.responses import JSONResponse 
from fastapi.middleware.cors import CORSMiddleware
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings 
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader,Webbased_Loader
from langchain.chains import ConversationalRetrievalChain
from langchain_community import Settings
from langchain.llms.openai import OpenAI
from langchain_openai import ChatOpenAI 
import awswrangler as wr 
import urllib.parse #manipulate urls
import boto3 #for working aws services
import uuid #for the universal unique ids
from typing import List, Optional,Tuple
import asyncio 
import uvicorn
from dotenv import load_dotenv
#optional imports
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings


load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv(OPENAI_API_KEY)


#declaring S3 bucket variables
#staging 
S3_BUCKET = "riddickbucket"
S3_REGION = "us-east-1"
S3_PATH = "documents_for_rag_app"

#aws bucket connection
aws_s3 = boto3.Session(aws_access_key_id = S3_KEY,
                       aws_secret_access_key = S3_SECRET,
                       region_name = S3_REGION,
                       )


try:
    #connect to the database
    MONGO_URL = "mongodb://localhost:27017/"
    client = pymongo.MongoClient(MONGO_URL, uuidRepresentation = 'standard')
    db = client['chat_with_doc']
    conversationalcol = db['chat_history']
    conversationalcol.create_index([("session_id")], unique = True)
    print('Debugging: ',conversationalcol)
except:
    #record and return detailed exception if any-with regards to connection
    #and index creation
    print(traceback.format_exc())
    exc_type, exc_obj, exc_tb = sys.exc_info()
    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    print(exc_type, fname, exc_tb.tb_lineno)

    

class ChatMessageSent(BaseModel):
    '''Pydantic model to specify session_id
       user_input and data_source
       allows the model to create these variables
       in the given structure and type hint
    '''
    session_id:str=None
    user_input: str
    data_source: str



def get_query_response(file_name:str,
                 session_id:str,
                 query:str,
                 model:str='gpt-3.5-turbo',
                 temperature:float=0.5):
    '''
    purpose: get response for a user query to the model
    args:(file_name)-name of uploaded file
    (session_id)-unique id for session for the user
    (query)-user query
    (model)-llm model to use for response generation
    (temperature)-response verbosity
    returns: conversational answer to user query
    '''
    print('filename:', file_name)
    filename = file_name.split('/')[-1]#ingnore title
    try:
        embeddings = OpenAIEmbeddings(model_name="text-embedding-3-small")
        Settings.embeddings=embeddings
    except Exception as e:
        print(f'Exception found: {e}')
    
    try:
        #required documents for context,loaded from connected s3 bucket
        wr.s3.download(path = f"s3://{S3_BUCKET}/{S3_PATH}/{filename}", local_file = filename, boto3_session = aws_s3)
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == '404':
            print(f'file/page not found: s3://{S3_BUCKET}/{S3_PATH}/{filename}')
        else:
            pass

    if filename.endswith(".docx"):
        loader = Docx2txtLoader(file_path = file_name.split('/')[-1])
    elif filename.endswith(".txt"):
        loader = TextLoader(file_path = file_name.split("/")[-1])
    else:
        loader = PyPDFLoader(file_name=file_name.split('/')[-1])

    try:
        data = loader.load()
        print("documents succefully loaded.....")
    except Exception:
        raise Exception('data not loaded successfully')
    
    #split document into nodes/chunks
    split_text = RecursiveCharacterTextSplitter(chunk_size = 1000,  chunk_overlap = 500, separators = ["\n", " ", ""])
    all_splits = split_text.split_documents(data)
    vectorstores = FAISS.from_documents(all_splits, embeddings)
    llm = OpenAI(model = model, temperature = temperature)
    Settings.llm = llm

    #create conversation chain-rag_chain:adds the chat_history,question,and retrieved ctx
    qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever = vectorstores.as_retriever())

    answer = qa_chain(
            {
            "question": query,
            "chat_history": get_chat_context(session_id = session_id),
            })

    return answer


def get_chat_context(session_id:str):
    '''
    Load conversation memory for specific session ids
    args: session_id-represents unique session id
    '''
    data = conversationalcol.find_one({"session_id": session_id}) 
    context = []
    if data:
        data = data['conversation']
        for x in range(0, len(data), 2):#step in 2s due to tuple of query and answer->(query,answer)#unbundled
            #get tuple if there is tuple to be gotten
            if x+1 < len(data):
                human = str(data[x]) if not isinstance(data[x], str) else data[x] 
                assistant = str(data[x + 1]) if not isinstance(data[x + 1], str) else data[x + 1]
                context.append((human, assistant))
    return context 


def get_session() -> str:
    '''create unique session id'''
    return str(uuid.uuid4())


def add_session_tochat_history(session_id:str, new_chat:List):
    '''extend the unique session id if any with new chat values'''
    chat_previous = conversationalcol.find_one({"session_id":session_id})
    if chat_previous == True:
        conversation_retrieved = chat_previous['conversation'] 
        conversation_retrieved.extend(new_chat)
        #update converation col for that session id in the database, ensure no duplicates
        conversationalcol.update_one({"session_id": session_id}, {"$set": {"conversation":conversation_retrieved}})
    else:
        conversationalcol.insert_one({"session_id": session_id, "conversation":new_chat})


#configure fastapi
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    #allow all connections
    allow_origins = ["*"],
    #allow all connection methods(GET,POST,...)
    allow_methods = ["*"],
    allow_credentials = False,
    allow_headers = ["*"],
    )


#set up chat endpoint-post method
@app.post("/chat")
async def create_chat_message(
    chats:ChatMessageSent):
    '''Endpoint to receive query from user and send to backend
    get_response function
    returns:json format object containng response and session_id(new/old)
    '''
    try:
        if chats.session_id is None:
            session_id = get_session()
            #create payload json object from chats
            #saves session id from get session, 
            payload = ChatMessageSent(
                session_id = session_id,
                user_input = chats.user_input,
                data_source = chats.data_source
                )
            payload = payload.model_dump()

            response = get_query_response(
                #returns 'answer'
                file_name = payload["data_source"],
                session_id = payload["session_id"],
                query = payload["user_input"]
                )
            
            add_session_tochat_history(session_id=session_id, 
                                       new_values = [(payload['user_input'],response['answer'])])
            return JSONResponse(content = {
                "response": response,
                "session_id": str(session_id)})
        else:
            payload = ChatMessageSent(
                session_id = str(chats.session_id),
                user_input = chats.user_input,
                data_source = chats.data_source)
            
            payload = payload.model_dump()
            
            response = get_query_response(
                file_name = payload.get("data_source",0),
                session_id = payload.get("session_id",0),
                query = payload.get("user_input",0),
                )
            
            add_session_tochat_history(session_id=str(chats.session_id), 
                                       new_values = [(payload.get('user_input'),response['answer'])])
            
            return JSONResponse(content = {
                "response":response,
                "session_id":str(chats.session_id) })
        
    except Exception:
        print(traceback.format_exc())
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        raise HTTPException(status_code = status.HTTP_204_NO_CONTENT, detail= 'Could not find content')

#file upload endpoint
@app.post("/uploadfile")
async def uploadtos3(data_file:UploadFile):
    print(data_file.filename.split("/")[-1])
    try:
        with open(f"{data_file.filename}", "wb") as out_file:
            content = await data_file.read()
            file_retrieved = out_file.write(content)
        wr.s3.upload(local_file = data_file.filename, path = f"s3://{S3_BUCKET}/{S3_PATH}/{data_file.filename.split("/")[-1]}",
                     boto3_session = aws_s3)
        os.remove(data_file.filename)
        os.remove(file_retrieved)

        response = {
            "filename":data_file.filename.split("/")[-1],
            "file_path": f"s3://{S3_BUCKET}/{S3_PATH}/{data_file.filename.split("/")[-1]}"
            }
            
    except FileNotFoundError:
        raise HTTPException(status_code = status.HTTP_404_NOT_FOUND, detail  = 'Item not found')
    return JSONResponse(content = response)        
            
if __name__ == "__main__":
    uvicorn.run(app)

    
        
        
        
        

            
        
    

    


