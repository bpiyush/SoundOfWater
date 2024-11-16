"""Utils for processing and encoding text."""

import torch



def lemmatize_verbs(verbs: list):
    from nltk.stem import WordNetLemmatizer
    wnl = WordNetLemmatizer()
    return [wnl.lemmatize(verb, 'v') for verb in verbs]


def lemmatize_adverbs(adverbs: list):
    from nltk.stem import WordNetLemmatizer
    wnl = WordNetLemmatizer()
    return [wnl.lemmatize(adverb, 'r') for adverb in adverbs]


class SentenceEncoder:

    def __init__(self, model_name="roberta-base"):
        from transformers import RobertaTokenizer, RobertaModel
        if model_name == 'roberta-base':
            self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
            self.model = RobertaModel.from_pretrained(model_name)
    
    def encode_sentence(self, sentence):
        inputs = self.tokenizer.encode_plus(
            sentence, add_special_tokens=True, return_tensors='pt',
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        # sentence_embedding = torch.mean(outputs.last_hidden_state, dim=1).squeeze(0)
        sentence_embedding = outputs.last_hidden_state[:, 0, :]
        return sentence_embedding
    
    def encode_sentences(self, sentences):
        """Encodes a list of sentences using model."""
        tokenized_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model(**tokenized_input)
        embeddings = outputs.last_hidden_state[:, 0, :]
        return embeddings
    