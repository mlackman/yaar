import pathlib 


def load_prompt(filename: str) -> str:
    file = pathlib.Path(__file__).parent / filename
    with file.open(mode='rt') as f:
        return f.read()
