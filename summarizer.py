from sumy.summarizers._summarizer import AbstractSummarizer
from sumy.models.dom._sentence import Sentence
from sumy._compat import Counter
import re

class RugbySummarizer(AbstractSummarizer):
    """
    Rugby Sports News Summarizer with keywords and tf features
    """

    def __call__(self, document, sentences_count):
        self._ensure_dependencies_installed()
        sentences_words = [self._to_word_set(s) for s in document.sentences]        
        if not sentences_words:
            return tuple()
        
        return self._compute_tf(sentences_words)


    @property
    def stop_words(self):
        return self._stop_words

    @stop_words.setter
    def stop_words(self, words):
        self._stop_words = frozenset(map(self.normalize_word, words))
    
    def _compute_tf(self, sentences):
        tf_values = map(Counter, sentences)

        tf_metrics = []
        for sentence in tf_values:
            metrics = {}
            max_tf = self._find_tf_max(sentence)

            for term, tf in sentence.items():
                metrics[term] = tf / max_tf

            tf_metrics.append(metrics)

        return tf_metrics

    @staticmethod
    def _find_tf_max(terms):
        return max(terms.values()) if terms else 1

    @staticmethod
    def _ensure_dependencies_installed():
        pass

    def _to_word_set(self, sentence):
        x_sentence = self._preprocess(sentence)
        words = map(self.normalize_word, x_sentence.words)
        return [self.stem_word(w) for w in words if w not in self._stop_words]

    def _preprocess(self, sentence):
        text = sentence._text
        text = re.sub(r'(\d{1,})[－−](\d{1,})', ' SCORE ', text)
        print(text)
        return Sentence(text, sentence._tokenizer)

from sumy.parsers.html import HtmlParser
from sumy.nlp.tokenizers import Tokenizer
import tinysegmenter
from sumy.utils import get_stop_words

LANGUAGE = "japanese"
SENTENCES_COUNT = 5

segmenter = tinysegmenter.TinySegmenter()

if __name__ == '__main__':
    url = "http://rugby-rp.com/news.asp?idx=111296&code_s=100310001006"
    parser = HtmlParser.from_url(url, Tokenizer(LANGUAGE))

    summarizer = RugbySummarizer()
    summarizer.stop_words = get_stop_words(LANGUAGE)

    summarizer(parser.document, 10)

