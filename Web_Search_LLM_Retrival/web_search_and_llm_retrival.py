# web_search_and_llm_retrival.py

from duckduckgo_search import DDGS
from openai import OpenAI
import httpx
import os

# ========== Client Setup ==========
client = OpenAI(
    base_url="http://localhost:12434/engines/v1",
    api_key="docker"
)

# Web search function using DuckDuckGo
def search_web_duckduckgo(query, max_results=5):
    results_text = ""
    with DDGS() as ddgs:
        results = ddgs.text(query, region='wt-wt', safesearch='Moderate', max_results=max_results)
        for r in results:
            results_text += f"{r.get('title')}\n{r.get('body')}\n{r.get('href')}\n\n"
    return results_text.strip()

# Function to query the LLM with the search results
def query_llm_with_search_results(question, search_results):
    response = client.chat.completions.create(
            model="ai/llama3.2:1B-Q4_0",
            messages=[
                {
                    "role": "system", 
                    "content": f"Use the following information from the web to answer the user's question.\n\nWeb Data:\n{search_results}\n\nQuestion: {question}"
                },
                {
                    "role": "user", 
                    "content": question
                }
            ],
            stream = True
        )
    print("\n📡 Answer:")
    for choice1 in response:
            content = getattr(choice1.choices[0].delta, "content", "")
            if content:
                print(content, end="", flush=True)
    print("\n" + "-"*50)

# ========== Main Function ==========
def main():
    print("🔍 Chat Search Bot (type 'exit' to quit)")
    while True:
        question = input("\n🧠 Your question: ")
        if question.lower() in ['exit', 'quit']:
            break
        print("🔎 Searching the web...")
        search_results = search_web_duckduckgo(question)
        # print(f"\n📄 Search Results Found:\n {search_results}")
        query_llm_with_search_results(question, search_results)

if __name__ == "__main__":
    main()