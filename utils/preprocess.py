def preprocess(text: str) -> str :
    """Preprocess one text"""
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = '[URL]' if t.startswith('http') else t
        new_text.append(t.strip('#'))
    return " ".join(new_text)

def batch_preprocess(texts: list) -> list :
    """Preprocess a group of words"""
    for idx in range(len(texts)):
        texts[idx] = preprocess(texts[idx])
    return texts
