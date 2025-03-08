from core.mind import Mind 
from interfaces.mother.mother_llm import MotherLLM 
from utils.llm_client import LLMClient 
from utils.tts_client import TTSClient 
 
def main(): 
    print("Initializing Large Mind Model...") 
    llm_client = LLMClient() 
    tts_client = TTSClient() 
    mother = MotherLLM(llm_client=llm_client, tts_client=tts_client) 
    mind = Mind() 
    mind.initialize_modules() 
    print("LMM initialized and ready for development") 
 
if __name__ == "__main__": 
    main() 
