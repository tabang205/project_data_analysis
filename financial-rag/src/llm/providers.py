from typing import Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from .base import BaseLLM


class TransformersLLM(BaseLLM):
    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen2.5-3B-Instruct",
        device: str = "cuda",
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        torch_dtype: str = "float16",
        load_in_4bit: bool = True
    ):
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        compute_dtype = torch.float16 if torch_dtype == "float16" and torch.cuda.is_available() else torch.float32

        quantization_config = None
        if load_in_4bit and device == "cuda":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True
            )

        print(f"[*] Đang nạp mô hình: {model_name_or_path} (4-bit={load_in_4bit})...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            quantization_config=quantization_config,
            device_map="auto" if device == "cuda" else None,
            torch_dtype=compute_dtype if not load_in_4bit else None
        )

        if device == "cpu":
            self.model.to("cpu")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> str:
        sys_msg = system_prompt or "You are a helpful assistant."
        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": prompt}
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        input_len = model_inputs.input_ids.shape[-1]

        target_max_tokens = max_new_tokens or self.max_new_tokens
        target_temp = self.temperature if temperature is None else temperature
        do_sample = target_temp > 0.0

        gen_kwargs = {
            "max_new_tokens": target_max_tokens,
            "pad_token_id": self.tokenizer.eos_token_id,
            "do_sample": do_sample
        }
        if do_sample:
            gen_kwargs["temperature"] = target_temp

        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                **gen_kwargs
            )

        new_tokens = generated_ids[0, input_len:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return response.strip()

