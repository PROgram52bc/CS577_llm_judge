import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
from dataclasses import dataclass

@dataclass
class PromptAugmentationConfig:
    force_answer: str = None
    ocr_augment: bool = False
    typos: bool = False
    non_influential: bool = False
    add_hyphens: bool = False
    non_unicode: bool = False
    synonyms: bool = False
    paraphrase: bool = False

class PromptAugmenter:
    """Class for augmenting the student answers in SciEntsBank"""
    def __init__(self, params: PromptAugmentationConfig):
        self.params = params
        self.nonUnicodeCandidates =[
            'é', 'ø', 'ü', 'ß',  # European characters
            '₽', '€', '¥', '£',  # Currency symbols
            '¿', '¡', '™', '©',  # Punctuation/symbols
            '∂', '∆', '∫', '∑'  # Math/Science symbols
        ]

    def run(self, text):
        if self.params.force_answer is not None:
            return self.params.force_answer
        if self.params.paraphrase:
            text = self.paraphrase_text(text)
        if self.params.synonyms:
            text = self.substitute_synonyms(text)
        if self.params.non_influential:
            text = self.add_non_influential(text)
        if self.params.typos:
            text = self.add_typos(text)
        if self.params.ocr_augment:
            text = self.ocr_augment(text)
        if self.params.non_unicode:
            text = self.add_non_unicode(text)
        if self.params.add_hyphens:
            text = self.add_hyphens(text)
        return text

    def ocr_augment(self, text):
        """Simulate text with OCR Errors"""
        aug = nac.OcrAug()
        augmented_texts = aug.augment(text)
        return augmented_texts[0]

    def add_typos(self, text):
        aug_keyboard = nac.KeyboardAug(aug_word_p=0.3, aug_char_p=.2)
        augmented_text = aug_keyboard.augment(text)
        return augmented_text[0]

    def add_non_influential(self, text):
        """Insert non-influential words into the text"""
        aug = naw.ContextualWordEmbsAug(
            model_path='bert-base-uncased',
            action="insert",
            aug_min=1,  # Insert at least 1 word
            aug_max=3,  # Insert at most 3 words
            aug_p=0.3,  # 30% chance of inserting a word at a random position
            device='cpu'  # Use 'cuda' if you have a compatible GPU
        )
        augmented_text = aug.augment(text)
        return augmented_text[0]

    def add_hyphens(self, text):
        """Return text with hyphens randomly inserted before each letter"""
        aug = nac.RandomCharAug(action="insert", candidates=["-"], aug_char_p=0.08)
        augmented_text = aug.augment(text)
        return augmented_text[0]

    def add_non_unicode(self, text):
        """Return text with non-unicode characters inserted into it"""
        aug = nac.RandomCharAug(action="insert",
                                aug_char_p= .1,
                                candidates=self.nonUnicodeCandidates)
        augmented_text = aug.augment(text)
        return augmented_text[0]

    def substitute_synonyms(self, text):
        """ Return text that has some words replaced with synonyms """
        aug = naw.ContextualWordEmbsAug(
            model_path='bert-base-uncased', action="substitute")
        augmented_text = aug.augment(text)
        return augmented_text[0]

    def paraphrase_text(self, text):
        """Returned paraphrased text by leveraging translation"""
        aug = naw.BackTranslationAug(
            from_model_name='Helsinki-NLP/opus-mt-en-de', # Found this model works better than default
            to_model_name='Helsinki-NLP/opus-mt-de-en'
        )
        augmented_text = aug.augment(text)
        return augmented_text[0]

