import os
import json
from langchain.memory import ConversationSummaryBufferMemory
from langchain.schema import HumanMessage, AIMessage, SystemMessage

HISTORY_FILE = "conversation_history.json"

class PersistentConversationSummaryMemory(ConversationSummaryBufferMemory):
    def __init__(self, session_id="default_session", llm=None, max_token_limit=3000):
        if llm is None:
            raise ValueError("ConversationSummaryBufferMemory requires an LLM for summarization.")
        
        super().__init__(
            llm=llm,
            max_token_limit=max_token_limit,
            return_messages=True,
            memory_key="history"
        )
        self.__dict__['session_id'] = session_id
        self.load_memory()

    def load_memory(self):
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                all_histories = json.load(f)
                session_data = all_histories.get(self.session_id, {})

                # Reconstruir mensajes
                serialized_msgs = session_data.get("messages", [])
                reconstructed_msgs = []
                for m in serialized_msgs:
                    if m["type"] == "human":
                        reconstructed_msgs.append(HumanMessage(**m))
                    elif m["type"] == "ai":
                        reconstructed_msgs.append(AIMessage(**m))
                    elif m["type"] == "system":
                        reconstructed_msgs.append(SystemMessage(**m))
                    else:
                        reconstructed_msgs.append(m)

                self.chat_memory.messages = reconstructed_msgs
                # Restaurar resumen si existe
                self.moving_summary_buffer = session_data.get("summary", "")
        else:
            self.chat_memory.messages = []
            self.moving_summary_buffer = ""

    def save_memory(self):
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                all_histories = json.load(f)
        else:
            all_histories = {}

        session_data = {
            "messages": [msg.dict() for msg in self.chat_memory.messages],
            "summary": self.moving_summary_buffer
        }
        all_histories[self.session_id] = session_data

        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(all_histories, f, ensure_ascii=False, indent=2)
    
    @property
    def session_id(self):
        return self.__dict__['session_id']