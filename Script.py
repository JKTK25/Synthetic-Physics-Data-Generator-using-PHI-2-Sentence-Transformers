import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import json
import random
from sentence_transformers import SentenceTransformer, util

# ✅ Load model and tokenizer
model_name = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# ✅ Load the sentence-transformer model for embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# ✅ Topics pool for variety
physics_topics = [
    "states of matter", "Newton's laws", "gravity", "thermodynamics",
    "electromagnetism", "light and optics", "quantum physics", "nuclear physics",
    "wave-particle duality", "forces and motion", "energy transformation", "electric circuits"
]

# ✅ Prompt Template with emoji integration
prompt_template = (
    "📘 Explain this physics concept in a simple way for students:\n"
    "Topic: {topic}\n"
    "Explanation:"
)

# ✅ Generation settings
batch_size = 1
max_new_tokens = 384
target_count = 1000
temperature = 0.9
top_p = 0.95

# ✅ Deduplication function using embeddings + cosine similarity
def deduplicate_embeddings(texts, threshold=0.92):
    embeddings = embedding_model.encode(texts, convert_to_tensor=True)
    unique_indices = []
    seen = torch.zeros(len(texts))

    for i in tqdm(range(len(texts)), desc="🧠 Filtering with Embeddings", dynamic_ncols=True):
        if seen[i]: continue
        unique_indices.append(i)
        sims = util.pytorch_cos_sim(embeddings[i], embeddings)[0]
        seen = seen | (sims > threshold)

    return [texts[i] for i in unique_indices]

# ✅ Output file
output_file = "synthetic_physics_data.jsonl"

# ✅ Open file for writing
generated_texts = []

with open(output_file, "w", encoding="utf-8") as f:
    for _ in tqdm(range(target_count), desc="🧪 Generating Physics Data", dynamic_ncols=True):
        topic = random.choice(physics_topics)

        prompt = prompt_template.format(topic=topic)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        # Remove prompt to keep only the explanation
        explanation = generated_text.split("Explanation:")[-1].strip()

        # Append the generated explanation for later deduplication
        generated_texts.append(explanation)

    # Deduplicate the generated texts using embeddings and cosine similarity
    unique_texts = deduplicate_embeddings(generated_texts)

    # Write the unique explanations to the file
    for text in unique_texts:
        json.dump({"text": text}, f)
        f.write("\n")

print("✅ Done generating and deduplicating synthetic physics data. The file is saved as 'synthetic_physics_data.jsonl'.")
