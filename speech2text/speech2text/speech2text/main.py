import torch
from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration
from datasets import load_dataset


MODEL="facebook/s2t-small-librispeech-asr"
PROCESSOR="facebook/s2t-small-librispeech-asr"
DATASET="hf-internal-testing/librispeech_asr_demo"

class speech2text:
    def __init__(self, data):
        self.model = Speech2TextForConditionalGeneration.from_pretrained(MODEL)
        self.processor = Speech2TextProcessor.from_pretrained(PROCESSOR)
        self.ds = load_dataset(data, "clean", split="validation")

    def transcribe(self):
        inputs = self.processor(self.ds[0]["audio"]["array"], sampling_rate=self.ds[0]["audio"]["sampling_rate"], return_tensors="pt")
        generated_ids = self.model.generate(inputs["input_features"], attention_mask=inputs["attention_mask"])
        transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        print(transcription)


speech2text(DATASET).transcribe()