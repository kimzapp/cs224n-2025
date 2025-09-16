from mteb.encoder_interface import PromptType
import numpy as np
import gensim
from mteb.models.wrapper import Wrapper

class Word2VecModel(Wrapper):
    def __init__(self, model_path: str, limit: int | None = None):
        super().__init__()
        self.model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True, limit=limit)

    def encode(
        self,
        sentences: list[str],
        task_name: str,
        prompt_type: PromptType | None = None,
        **kwargs,
    ) -> np.ndarray:
        embeddings = []
        for sentence in sentences:
            words = sentence.split()
            word_vectors = [self.model.get_vector(word) for word in words if word in self.model]
            if word_vectors:
                sentence_embedding = np.mean(word_vectors, axis=0)
            else:
                sentence_embedding = np.zeros(self.model.vector_size)
            embeddings.append(sentence_embedding)
        return np.array(embeddings)