import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

class MedicalChatbot:
    def __init__(self, model_name: str = "meta-llama/Llama-2-7b-chat-hf"):
        """
        Initializes the chatbot by loading the Llama model and tokenizer.
        :param model_name: Name of the pre-trained Llama model to use.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = LlamaTokenizer.from_pretrained(model_name)
        self.model = LlamaForCausalLM.from_pretrained(model_name).to(self.device)

    def generate_response(self, user_input: str, max_length: int = 200):
        """
        Generates a response from the chatbot based on user input.
        :param user_input: The text input from the user.
        :param max_length: Maximum length of the generated response.
        :return: The chatbot's response as a string.
        """
        inputs = self.tokenizer(user_input, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_length=max_length)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

# Example usage
if __name__ == "__main__":
    chatbot = MedicalChatbot()
    user_query = "What are the symptoms of diabetes?"
    print("Chatbot:", chatbot.generate_response(user_query))
