def preprocess(text: str) -> str :
    text = '@user' if text.startswith('@') and len(text) > 1 else text
    text = '[URL]' if text.startswith('http') else text
    return text.strip('#')

def text_process(text: str) -> str :
    new_text = []
    text = text.replace("\n", " ")
    for t in text.split(" "):
        t = preprocess(t)
        nbsp_pos = t.find("\xa0")
        while nbsp_pos != -1:
            if t[nbsp_pos + 1:].startswith('@') and len(t[nbsp_pos + 1:]) > 1:
                new_text.append(preprocess(t[: nbsp_pos]))
                t = t[nbsp_pos + 1:]
                nbsp_pos = t.find("\xa0")
                t = '@user' if nbsp_pos == -1 else t
            elif t[nbsp_pos + 1:].startswith('http'):
                new_text.append(preprocess(t[: nbsp_pos]))
                t = t[nbsp_pos + 1:]
                nbsp_pos = t.find("\xa0")
                t = '[URL]' if nbsp_pos == -1 else t
            elif t[nbsp_pos + 1:].startswith('#'):
                new_text.append(preprocess(t[: nbsp_pos]))
                t = t[nbsp_pos + 1:]
                nbsp_pos = t.find("\xa0")
                t = t.strip('#') if nbsp_pos == -1 else t
            else:
                nbsp_pos = t.find("\xa0", nbsp_pos + 1)
        new_text.append(t)
    return " ".join(new_text)