from ctransformers import AutoModelForCausalLM
from retrieval import Retrieval
import time

class Response_Generator():
    def __init__(self):
        print('Initialize llm and retrieval module')
        self.llm = AutoModelForCausalLM.from_pretrained(
                'models/vi-LLM/ggml-vistral-7B-chat-q5_1.gguf',
                model_type='mistral',
                max_new_tokens = 1000,
                context_length = 6000,
            )
        self.retrieval = Retrieval()
        print('Done! Ready to use')
    def llm_response(self, user_query: str):
        context = self.retrieval.db_query(user_query)
        prompt = """
[INST] Bạn là một trợ lí Tiếng Việt nhiệt tình và trung thực. Hãy luôn trả lời một cách hữu ích nhất có thể, đồng thời giữ an toàn.
Cho đoạn thông tin sau hãy dựa vào thông tin đó để trả lời những câu hỏi của người dùng:
{context}
Câu hỏi: {query} ?
[/INST]
""".format(context=context, query=user_query)
        respone = self.llm(prompt)
        return respone

# my_synthsis = Response_Generator()
# query = 'nguyên nhân bị trĩ và cách điều trị'
# print(my_synthsis.llm_response(query))
