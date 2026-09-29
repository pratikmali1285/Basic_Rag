"""Step 6: connect to the LLM (the "brain" of the assistant)."""


from langchain_groq import ChatGroq

from hr_assistant import config
from hr_assistant.gateway import get_gateway_llm

from hr_assistant.logger import get_logger

logger = get_logger(__name__)

def get_llm():
    """Return the chat model: via Portkey if PORTKEY_API_KEY is set, else Groq directly."""
    if config.USE_PORTKEY:
        logger.info("Initializing LLM via Portkey")
        return get_gateway_llm()

    logger.info("PORTKEY_API_KEY not set, calling Groq directly")
    return ChatGroq(model=config.LLM_MODEL_NAME, api_key=config.GROQ_API_KEY)
