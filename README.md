
# ⚛️ Synthetic Physics Data Generator using PHI-2 + Sentence Transformers

This Python script generates **simple, student-friendly explanations** of various physics concepts using a powerful language model (`phi-2`) and filters them using sentence embeddings to ensure **uniqueness and quality**. It’s perfect for educators, content creators, and researchers looking to build datasets or educational material.

---

## 📦 Features

- **Text Generation with PHI-2**: Uses Microsoft's `phi-2` model to generate natural, easy-to-understand explanations.
- **Semantic Deduplication**: Uses `MiniLM` sentence embeddings to remove semantically similar outputs.
- **Batch Generation**: Creates up to `1000` explanations with random topics for variety.
- **Clean Output**: Saves all unique explanations in JSONL format for easy downstream use.

---

## 🧪 Requirements

Make sure you have Python 3.8+ and install the following dependencies:

```bash
pip install torch transformers sentence-transformers tqdm
```

> For CUDA/GPU support, ensure PyTorch is installed with CUDA version compatible with your system.

---

## ⚙️ How to Use

1. **Clone or Download the Script**

2. **Run the script** using:

```bash
python synthetic_physics_generator.py
```

This will:

- Load the `phi-2` model and tokenizer
- Generate 1000 physics concept explanations (you can customize this)
- Deduplicate them using cosine similarity between sentence embeddings
- Save the results in a file named: `synthetic_physics_data.jsonl`

---

## 📝 Output Format

Each line in the output file is a JSON object:

```json
{"text": "Matter exists in three main states: solid, liquid, and gas. ..."}
```

You can load this in Python as follows:

```python
import json

with open("synthetic_physics_data.jsonl", "r") as f:
    data = [json.loads(line) for line in f]
```

---

## 🧠 Topics Covered

Randomly chosen from a diverse pool including:

- States of matter
- Newton's Laws
- Gravity
- Thermodynamics
- Electromagnetism
- Light and optics
- Quantum physics
- Nuclear physics
- Wave-particle duality
- Forces and motion
- Energy transformation
- Electric circuits

Feel free to edit the `physics_topics` list to focus on specific domains!

---

## ✨ Customization Tips

- Change `target_count` to generate more or fewer entries.
- Modify `prompt_template` to adjust the tone or format of the prompt.
- Tweak `temperature` and `top_p` for more creative or conservative generation.

---

## ⚠️ Warnings & Recommendations

- This script is **resource-intensive**. Running it on CPU may be slow.
- Make sure to run in a virtual environment or Docker to avoid version conflicts.
- PHI-2 is for **research** use only; verify licensing before using in production.

---

## 🙌 Acknowledgments

- [Microsoft](https://huggingface.co/microsoft/phi-2) for the `phi-2` model
- [Hugging Face](https://huggingface.co/) for `transformers` and `sentence-transformers`
- [TQDM](https://github.com/tqdm/tqdm) for progress bars

---

## 📂 License

This code is licensed under the MIT License. Please review model licenses before use.

---

## ✉️ Contact

For feedback or collaboration, feel free to reach out!

**Author**: *Kariuki James*  
**Email**: jamexkarix54@gmail.com  
**Phone**: 0718845849

---

Happy generating! ⚡
