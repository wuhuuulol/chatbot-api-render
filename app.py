import os
from flask import Flask, request, jsonify
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from flask_cors import CORS  # CORS desteği ekleyin

app = Flask(__name__)
CORS(app)  # CORS'u etkinleştirin

# Bu fonksiyonu Gemini API anahtarınızı saklamak için kullanabilirsiniz
def get_api_key():
    # Güvenlik için çevresel değişkenlerden API anahtarını alın
    api_key = os.environ.get("AIzaSyD0TJxBpLirLbDfhgGTsfstG4gU9E3_lAA")
    if not api_key:
        # Eğer çevresel değişken yoksa, bir dosyadan okuyabilirsiniz
        try:
            with open("api_key.txt", "r") as f:
                api_key = f.read().strip()
        except:
            api_key = "AIzaSyD0TJxBpLirLbDfhgGTsfstG4gU9E3_lAA"
    return api_key

# Global değişkenler
system_prompt = "her mesajında en fazla 3 kelime kullan"  # Varsayılan prompt
chat_instances = {}

# API anahtarını ayarla
os.environ["GOOGLE_API_KEY"] = get_api_key()

@app.route('/set_system_prompt', methods=['POST'])
def set_system_prompt():
    global system_prompt
    data = request.json
    new_system_prompt = data.get('system_prompt', '')
    
    if new_system_prompt:
        system_prompt = new_system_prompt
        
        # Tüm mevcut sohbet örneklerini sıfırla
        global chat_instances
        chat_instances = {}
        
        print(f"Sistem prompt güncellendi: {system_prompt}")
        return jsonify({"status": "success", "system_prompt": system_prompt})
    else:
        return jsonify({"status": "error", "message": "Sistem promptu boş olamaz"}), 400

@app.route('/get_system_prompt', methods=['GET'])
def get_system_prompt():
    global system_prompt
    return jsonify({"system_prompt": system_prompt})

@app.route('/chat', methods=['POST'])
def chat():
    global system_prompt
    data = request.json
    message = data.get('message', '')
    session_id = data.get('session_id', 'default')
    
    print(f"Gelen mesaj: {message}, Oturum ID: {session_id}")
    print(f"Kullanılan sistem prompt: {system_prompt}")
    
    # Eğer bu oturum için bir sohbet örneği yoksa, yeni bir tane oluştur
    if session_id not in chat_instances:
        try:
            # Sistem promptu kullanarak bir chatbot oluştur
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}")
            ])
            
            # Gemini 2.0 Flash Lite modeli
            llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite")
            
            # Konuşma geçmişini hatırla
            memory = ConversationBufferMemory(return_messages=True, memory_key="history")
            
            # Konuşma zinciri oluştur
            chat_chain = ConversationChain(
                llm=llm,
                prompt=prompt,
                memory=memory,
                verbose=True
            )
            
            chat_instances[session_id] = chat_chain
            print(f"Yeni sohbet örneği oluşturuldu. Sistem prompt: {system_prompt}")
        except Exception as e:
            print(f"Hata oluştu: {str(e)}")
            return jsonify({"error": f"Sohbet örneği oluşturulurken hata: {str(e)}"}), 500
    
    # Var olan sohbet örneğini kullan
    chat_chain = chat_instances[session_id]
    
    try:
        # Kullanıcı mesajını işle ve yanıt al
        response = chat_chain.predict(input=message)
        print(f"Yanıt: {response}")
        return jsonify({"response": response})
    except Exception as e:
        print(f"Yanıt alınırken hata: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/reset', methods=['POST'])
def reset_chat():
    data = request.json
    session_id = data.get('session_id', 'default')
    
    if session_id in chat_instances:
        del chat_instances[session_id]
        print(f"Sohbet sıfırlandı: {session_id}")
    
    return jsonify({"status": "success", "message": "Chat session reset"})

if __name__ == '__main__':
    print(f"API başlatıldı. Varsayılan sistem prompt: {system_prompt}")
    app.run(host='0.0.0.0', port=5000, debug=True)