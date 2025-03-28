import os

def get_save_path(root_dir, prefix='run_'):
    """
    Generate the next non-duplicate subdirectory
    """
    os.makedirs(root_dir, exist_ok=True)
    existing = [d for d in os.listdir(root_dir) if d.startswith(prefix)]
    run_ids = sorted([
        int(d.replace(prefix, ''))
        for d in existing if d.replace(prefix, '').isdigit()
    ])
    next_id = (run_ids[-1] + 1) if run_ids else 1
    new_path = os.path.join(root_dir, f"{prefix}{next_id}")
    os.makedirs(new_path, exist_ok=True)
    return new_path