import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict

from configs.base_config import DEVICE, WEIGHTS_DIR
from configs.stage3_config import (
    VLM_MODEL_NAME,
    LOAD_IN_4BIT,
    LOAD_IN_8BIT,
    LORA_RANK,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_TARGET_MODULES,
    LORA_WEIGHTS_PATH,
    SYSTEM_PROMPT_CAPTION,
    VQA_QUESTIONS,
    MAX_NEW_TOKENS,
    TEMPERATURE,
    TOP_P,
    DO_SAMPLE,
)


class InternVL2Wrapper:
    """
    Wrapper cho InternVL2.
    """

    def __init__(
        self,
        device: str = DEVICE,
        use_lora: bool = True,
        lora_weights: str = None,
    ):
        self.device = device
        self.use_lora = use_lora

        self.model = None
        self.tokenizer = None

        self._load_model()

        if use_lora:
            self._inject_lora()

        weights_path = lora_weights or LORA_WEIGHTS_PATH

        if Path(weights_path).exists():
            self._load_lora_weights(weights_path)

    def _load_model(self):

        try:
            from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
        except ImportError:
            raise ImportError("pip install transformers peft accelerate bitsandbytes")

        cache_dir = Path(WEIGHTS_DIR) / "internvl2"

        if cache_dir.exists():
            model_id = str(cache_dir)
        else:
            model_id = VLM_MODEL_NAME

        print(f"[InternVL2] Loading model: {model_id}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            use_fast=False,
        )

        # Chọn quantization theo config
        if LOAD_IN_4BIT:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,  # bfloat16 ổn định hơn float16
                bnb_4bit_use_double_quant=True,
            )
        elif LOAD_IN_8BIT:
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            bnb_config = None

        load_kwargs = dict(
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        if bnb_config is not None:
            load_kwargs["quantization_config"] = bnb_config

        self.model = AutoModel.from_pretrained(model_id, **load_kwargs)
        self.model.eval()

        print(f"[InternVL2] Model loaded | 4bit={LOAD_IN_4BIT} 8bit={LOAD_IN_8BIT}")

    def _inject_lora(self):

        try:
            from peft import (
                get_peft_model,
                LoraConfig,
                TaskType,
                prepare_model_for_kbit_training,
            )
        except ImportError:
            raise ImportError("pip install peft")

        print("[InternVL2] Injecting LoRA into language_model...")

        # Apply LoRA chỉ vào language_model bên trong InternVL2
        # Tránh lỗi inputs_embeds khi wrap toàn bộ InternVLChatModel với PEFT
        lm = self.model.language_model

        if LOAD_IN_4BIT or LOAD_IN_8BIT:
            lm = prepare_model_for_kbit_training(lm, use_gradient_checkpointing=False)

        lora_cfg = LoraConfig(
            r=LORA_RANK,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=LORA_TARGET_MODULES,
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        )

        self.model.language_model = get_peft_model(lm, lora_cfg)

        trainable = sum(p.numel() for p in self.model.language_model.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.model.parameters())

        print(
            f"[InternVL2] LoRA injected | "
            f"trainable: {trainable:,} / {total:,} "
            f"({100 * trainable / total:.2f}%)"
        )

    def _count_params(self):
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.model.parameters())
        return trainable, total

    def _load_lora_weights(self, path: str):
        try:
            from peft import PeftModel
            self.model.language_model = PeftModel.from_pretrained(
                self.model.language_model, path
            )
            print(f"[InternVL2] LoRA weights loaded ← {path}")
        except Exception as e:
            print(f"[InternVL2] Warning: {e}")

    def save_lora(self, path: str = None):
        save_path = path or LORA_WEIGHTS_PATH
        Path(save_path).mkdir(parents=True, exist_ok=True)
        self.model.language_model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"[InternVL2] LoRA saved → {save_path}")

    def _build_pixel_values(self, image_np):

        from torchvision import transforms
        from PIL import Image

        transform = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        image = Image.fromarray(image_np)

        pixel_values = transform(image).unsqueeze(0)

        pixel_values = pixel_values.to(
            dtype=torch.bfloat16,
            device=self.device,
        )

        return pixel_values

    @torch.no_grad()
    def caption(
        self,
        image_np,
        extra_context: str = "",
    ) -> str:

        pixel_values = self._build_pixel_values(image_np)

        prompt = SYSTEM_PROMPT_CAPTION

        if extra_context:
            prompt += f"\nThông tin bổ sung: {extra_context}"

        prompt += "\n<image>\nMô tả lỗi:"

        generation_config = dict(
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=DO_SAMPLE,
            temperature=TEMPERATURE if DO_SAMPLE else None,
            top_p=TOP_P if DO_SAMPLE else None,
        )

        generation_config = {
            k: v
            for k, v in generation_config.items()
            if v is not None
        }

        response = self.model.chat(
            self.tokenizer,
            pixel_values,
            prompt,
            generation_config,
        )

        return response.strip()

    @torch.no_grad()
    def vqa(
        self,
        image_np,
        questions: Dict[str, str] = None,
    ) -> Dict[str, str]:

        pixel_values = self._build_pixel_values(image_np)

        qs = questions or VQA_QUESTIONS

        answers = {}

        generation_config = dict(
            max_new_tokens=64,
            do_sample=False,
        )

        for key, question in qs.items():

            prompt = f"<image>\n{question}"

            response = self.model.chat(
                self.tokenizer,
                pixel_values,
                prompt,
                generation_config,
            )

            answers[key] = response.strip()

        return answers

    def get_trainable_params(self):
        return [p for p in self.model.language_model.parameters() if p.requires_grad]

    def train_mode(self):

        self.model.train()

    def eval_mode(self):

        self.model.eval()


if __name__ == "__main__":

    print("=== InternVL2Wrapper Test ===")

    model = InternVL2Wrapper(
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_lora=False,
    )

    trainable, total = model._count_params()

    print(f"Trainable params: {trainable:,}")
    print(f"Total params: {total:,}")

    if torch.cuda.is_available():

        print(
            f"VRAM used: "
            f"{torch.cuda.memory_allocated()/1e9:.2f} GB"
        )