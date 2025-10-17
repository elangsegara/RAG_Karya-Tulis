from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    google_api_key="AIzaSyDkIz_8NbB_cqzCOzO45LIEnFiznd3CTac"  # Replace with your actual API key
)

print("Successfully created Gemini LLM!")

response = llm.invoke("Hello Gemini! How are you today?")
print("Model response:", response.content)

