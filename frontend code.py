import requests
import json
import streamlit as st
import time
import os
import warnings
warnings.filterwarnings('ignore')
BACKEND_URL = "http://127.0.0.1:8000"

def chat(user_input, data, session_id=None):
    url = f"{BACKEND_URL}/chat"
    print('user:', user_input)
    print('data:', data)
    print('session_id:', session_id)

    if session_id is None:
        payload = json.dumps({'user_input':user_input, 'data_source':data})
    else:
        payload = json.dumps({'user_input':user_input, 'data_source':data, 'session_id':session_id})
    #set headers for api request
    headers = {'accept': 'application/json',
               'content_Type':'application/json'}
    #make a postrequest to api chat endpoint
    response = requests.request('POST', url, headers=headers, data=payload)
    print(response.json())

    #checking status of connection
    if response.status_code == 200:
        return response.json()['response']['answer'], response.json()['session_id'] #collect from the backend response

def upload_file(file_path):
    url = f"{BACKEND_URL}/uploadfile"
    print(url)
    print('file path:', file_path)

    file_name = file_path.split("/")[-1]

    payload = {}
    files = [('data_file',(file_name, open(file_path, 'rb'), 'application/pdf'))]
    headers = {'accept': 'application/json'}
    response = requests.request('POST', url, headers = headers, data = payload, files = files)
    print(response.status_code)

    if response.status_code == 200:
        print(response.json())
        return response.json()['file_path']

st.set_page_config(page_title='Document Chat', layout='wide', page_icon='🧠')

st.title("🧾 Document Chatbot")

if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'sessionid' not in st.session_state:
    st.session_state.sessionid = None

data_file = st.file_uploader(
    label = 'Input File', accept_multiple_files = False, type = ['pdf', 'docx'])
st.divider()
if data_file is not None:
    temp_dir = os.path.join(os.getcwd(), 'temp')
    os.makedirs(temp_dir, exist_ok= True)
    file_path = os.path.join(temp_dir,data_file.name)
    with open(file_path, 'wb') as f:
        f.write(data_file.getbuffer())
    s3_upload_url = upload_file(file_path = file_path)
    s3_upload_url = s3_upload_url.split("/")[-1]
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

    prompt = st.chat_input('You can ask any question')
    if prompt:
        st.session_state.messages.append(
        {'role':'user', 'content':prompt}
        )
        with st.chat_message('user'):
            st.markdown(prompt)
        with st.chat_message('assistant'):
            if st.session_state.sessionid is None:
                assistant_response, session_id = chat(prompt, data=s3_upload_url, session_id = None)
                st.session_state.sessionid = session_id

            else:
                assistant_response, session_id = chat(
                    prompt, data = s3_upload_url, session_id = st.session_state.sessionid)
            message_placeholder = st.empty()
            full_response = " "

            for chunk in assistant_response.split():
                full_response += chunk + " "
                time.sleep(0.05)
                message_placeholder.markdown(full_response + "_")
            message_placeholder.markdown(full_response)
        st.session_state.messages.append({
            "role":"assistant", "content":full_response}
                                     )
            
        
            
    
        
            
