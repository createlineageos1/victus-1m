# READ THIS FOR A BIT
You are now in candy branch, where Victus-1M has a bigger dataset.
The loss can be higher on this branch.

# Victus 1M
**Victus 1M** is a minimalist math-focused language model built for educational and experimental purposes. It runs on both mobile and desktop platforms with minimal resource usage and no reliance on massive libraries.

> ⚠️ **Disclaimer:** This project is **not related to cybersecurity**, hacking, penetration testing, or reverse engineering. It is purely a lightweight AI chatbot for solving simple math expressions.

---

## 🎯 What is this?

victus-1m is:

- A **tiny Transformer-based model** for math problem solving.
- Built to be **portable and lightweight** (e.g. for Android, Raspberry Pi, Hexrail or old PCs).
- Designed to **answer simple math questions**, like `2+3`, `9/3`, or `what is 7 times 8`.
- Fully **offline and privacy-friendly**.

---

## 🔧 Features

- 🧠 Minimal Transformer model (PyTorch)
- 🧮 Symbolic math parser using safe Python evaluation
- 🛠️ Runs with only `torch`
- 📱 Compatible with mobile Python environments (like Pydroid3 or Termux)
- 💡 Easy to understand, hack, and modify

---

## 🛠 Requirements

- Python 3.7+
- `torch`

Install with:

```bash pip install torch```

## 📁 Project Structure
main.py: Main script to run the chatbot

## 🧠 How It Works
Uses Python’s eval() safely to compute basic expressions like 3+2, 5*7.

If natural language input is detected, the transformer attempts to complete or generate an answer (e.g., 2+2).

Falls back to symbolic methods if inference is not required.

## ❌ Not Included
No LLM features like reasoning, memory, or context understanding

No integration with web, database, or cloud services

No relation to hacking, malware, or offensive security tools

## 📚 Example Use Cases
As a lightweight offline math tutor

As a toy AI project on embedded devices

For learning the basics of Transformer models

For math-based CLI tools

## 🧪 Testing
We are testing and training our models in Google Colab with v2-8 TPU and 330 GB of RAM capacity

## 📜 License
MIT License. Open-source and free to use in personal or educational projects

## 🙋‍♂️ Creator
Built by Alexander/Nuri – follow for more weird and lightweight AI stuff
