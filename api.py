from fastapi import FastAPI
from pydantic import BaseModel
from transformers import PegasusTokenizer, TFPegasusForConditionalGeneration


class Summaries(BaseModel):
    original_text: str
    summary: str


def summarizing_model(text):
    model_name = "human-centered-summarization/financial-summarization-pegasus"
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = TFPegasusForConditionalGeneration.from_pretrained(model_name)

    input_ids = tokenizer(text, return_tensors="tf").input_ids

    output = model.generate(
        input_ids,
        max_length=32,
        num_beams=5,
        early_stopping=True
    )

    return str(tokenizer.decode(output[0], skip_special_tokens=True))


summary_api = FastAPI()


@summary_api.post("/fetch_summaries/")
async def create_summary(summary: Summaries):
    summary_dict = summary.dict()
    print(summary_dict)
    result = summarizing_model(summary_dict['original_text'])
    summary_dict.update({"summary": str(result)})
    return summary_dict


@summary_api.get("/")
async def root():
    return {"message": "Welcome to Financial Summarizing API"}
