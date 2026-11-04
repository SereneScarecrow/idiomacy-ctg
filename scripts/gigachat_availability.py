import os
from dotenv import load_dotenv
from gigachat import GigaChat

load_dotenv()

giga = GigaChat(
   credentials=os.getenv("GIGACHAT_API_KEY"),
   verify_ssl_certs=False
)

response = giga.get_models()

print(response)