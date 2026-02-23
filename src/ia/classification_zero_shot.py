from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

""" https://huggingface.co/deepseek-ai/deepseek-coder-1.3b-instruct """
MODEL_NAME = "deepseek-ai/deepseek-coder-1.3b-instruct"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(
    "deepseek-ai/deepseek-coder-1.3b-instruct",
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/deepseek-coder-1.3b-instruct",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
).cuda()


def zero_shot_predict_sentiment(review_text):
    prompt = [
        {'role': 'system',
         'content': f"""
            You are a strict sentiment classifier.
            
            Rules:
            - If the review expresses strong satisfaction, praise, recommendation → Positive
            - If the review expresses dissatisfaction, complaint, anger → Negative
            - If the review is mixed or neutral → Neutral
            
            Respond with ONLY one word:
            Positive
            Negative
            Neutral
            
            Now classify:
            Review: """ + review_text + """
            
            Sentiment:
            """}
    ]

    inputs = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, return_tensors="pt").to(device)

    outputs = model.generate(
        inputs,
        max_new_tokens=4,
        do_sample=False,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
    )

    return tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)


if __name__ == "__main__":
    print(zero_shot_predict_sentiment(
        "I absolutely loved this product! It exceeded my expectations and I would highly recommend it to anyone."))
