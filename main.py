from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama


import os
load_dotenv()

def main():
    print("Hello from langchain-course!")
    print(os.environ.get("OPENAI_API_KEY"))
    information = """
McLaren Automotive replaced McLaren Cars in 2010. McLaren Cars had been founded in 1985[5] and released the McLaren F1 in 1992. Between 1994 and 2010, McLaren Cars was registered as a 'dormant company', before the founding of McLaren Automotive in 2010. The new company was originally separate from the existing McLaren companies to enable investment in the new venture, but was brought together in July 2017 after Ron Dennis sold his shares in McLaren Automotive and McLaren Group.

McLaren's Formula One founder Bruce McLaren was born in Auckland, New Zealand in 1937,[6] and learned about cars and engineering at his parents' service station and workshop there. By 15, he had entered a local hillclimb in an Austin 7 Ulster, winning his first race in the car.[7] In 1958, McLaren arrived in the United Kingdom with the 'Driver to Europe' scheme, intended to help Australian and New Zealand racers to compete in Europe. His mentor, Jack Brabham, introduced him to Cooper Car Company, a small team based in Surbiton, Surrey. Auspiciously starting his Formula One career in 1958, McLaren joined the Formula One team a year later. That same year, he won the US Grand Prix at age 22, making him the youngest Grand Prix winner to that date.[8] He stayed with Cooper for a further seven years, winning three more Grands Prix and other races, driving for Jaguar and Aston Martin, and winning the 24 Hours of Le Mans in 1966 with Ford.
"""
    summary_template = """
    Given the following information {information} about Mclaren, do the followings:
    1. Summarize the information in a concise manner.
    2. Provide 2 interesting facts about Mclaren.
    """
    
    summary_prompt_template = PromptTemplate(input_variables=["information"], template=summary_template)

    #llm = ChatOpenAI(model_name="gpt-5", temperature=0)
    llm = ChatOllama(model = "gemma3:270m", temperature=0.3)

    chain =    summary_prompt_template | llm
    response = chain.invoke({"information": information})

    print(response.content)


if __name__ == "__main__":
    main()
