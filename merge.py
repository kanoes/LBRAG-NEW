from pathlib import Path

def merge_code_files():
    script_dir = Path(__file__).resolve().parent
    lbrag_dir = script_dir / "lbrag"
    output_file = script_dir / "code.txt"
    experiment_file = script_dir / "experiments" / "scripts" / "experiment.py"
    prepare_samples_file = script_dir / "experiments" / "scripts" / "prepare_mkqa_samples.py"
    
    if not lbrag_dir.exists():
        return
    
    py_files = sorted([f for f in lbrag_dir.glob("*.py") if f.name != "__pycache__"])
    
    if not py_files:
        return
    
    content_parts = []
    
    for py_file in py_files:
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                file_content = f.read()
            content_parts.append(f"<{py_file.name}>\n{file_content}")
        except Exception:
            continue
    
    if experiment_file.exists():
        try:
            with open(experiment_file, 'r', encoding='utf-8') as f:
                experiment_content = f.read()
            content_parts.append(
                f"<experiments/scripts/experiment.py>\n{experiment_content}"
            )
        except Exception:
            pass
    
    if prepare_samples_file.exists():
        try:
            with open(prepare_samples_file, 'r', encoding='utf-8') as f:
                prepare_samples_content = f.read()
            content_parts.append(
                f"<experiments/scripts/prepare_mkqa_samples.py>\n{prepare_samples_content}"
            )
        except Exception:
            pass
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n\n'.join(content_parts))
    except Exception as e:
        print(e)

if __name__ == "__main__":
    merge_code_files()
