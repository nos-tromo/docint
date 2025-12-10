from pathlib import Path


project_root: Path = Path(__file__).parents[2].resolve()
print(project_root)
