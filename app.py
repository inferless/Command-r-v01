import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"]='1'
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM , BitsAndBytesConfig

class InferlessPythonModel:
    def initialize(self):
        model_id = "CohereForAI/c4ai-command-r-v01"
        snapshot_download(repo_id=model_id,allow_patterns=["*.safetensors"])
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_id,quantization_config=bnb_config,device_map = 'cuda')

    def infer(self,inputs):
        prompt = inputs["prompt"]
        messages = [{"role": "user", "content": prompt}]
        input_ids = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
        generated_tokens = self.model.generate(input_ids,max_new_tokens=256,do_sample=True,temperature=0.1)
        generated_text = self.tokenizer.batch_decode(generated_tokens[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
        return { "generated_text" : generated_text}

    def finalize(self):
        self.pipe = None
