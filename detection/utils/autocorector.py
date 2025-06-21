from textblob import Word

def autocorrect(sentence: str, threshold=.76) -> str:
    transformed_sentence = sentence.split(" ")
    words = [Word(word) for word in transformed_sentence]
    autocorrected_words = []

    for word in words:
        w = word.spellcheck()
        k = max(w, key=lambda x: x[1] >= threshold)
        autocorrected_words.append(k[0])
    
    return " ".join(autocorrected_words)

