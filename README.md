# Conversational Retrieval Agent

This is an end-to-end example of a conversational retrieval agent for chatting with LangSmith documentation.
For more information on conversational retrieval agents, see [this documentation](https://python.langchain.com/docs/use_cases/question_answering/how_to/conversational_retrieval_agents).

To setup: `pip install -r requirements.txt`

To run: `streamlit run streamlit.py`

This relies on:

- OpenAI for embedding and language model
- [Streamlit](https://github.com/langchain-ai/streamlit-agent) for the UI
- [LangSmith](https://docs.smith.langchain.com/) for logging feedback
