# Sample data layout (Bias in Bios)

The full **Bias in Bios** dataset is **not** stored in this repository (too large). It is downloaded automatically on first use from Hugging Face: [`LabHC/bias_in_bios`](https://huggingface.co/datasets/LabHC/bias_in_bios).

`bias_in_bios_example.json` shows the **fields** and **two short synthetic rows** (placeholders) so graders and collaborators can see the schema without the full corpus.

**Preprocessing** (in code): `data/bias_in_bios.py` tokenizes biographies with the Qwen tokenizer, uses predefined train/validation/test splits, and aligns `labels` (occupation id, 0–27) with `gender` (sensitive attribute for probing).
