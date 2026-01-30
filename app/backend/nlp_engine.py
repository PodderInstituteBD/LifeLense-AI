python
"""
Open-source NLP engine for text processing.
No external APIs - all logic is local.
"""

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import re
from typing import List, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
import logging

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

logger = logging.getLogger(__name__)

class TextAnalyzer:
    """Core NLP engine for text analysis"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep punctuation for sentence splitting
        text = re.sub(r'[^\w\s\.\!\?\-\,]', '', text)
        return text.strip()
    
    def extract_sentences(self, text: str) -> List[str]:
        """Extract sentences from text"""
        try:
            sentences = sent_tokenize(text)
            # Filter out very short sentences
            sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
            return sentences
        except Exception as e:
            logger.error(f"Error extracting sentences: {e}")
            return []
    
    def extract_keywords(self, text: str, top_n: int = 10) -> List[Tuple[str, float]]:
        """Extract keywords using TF-IDF"""
        try:
            sentences = self.extract_sentences(text)
            if not sentences:
                return []
            
            vectorizer = TfidfVectorizer(
                max_features=100,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            try:
                X = vectorizer.fit_transform(sentences)
                feature_names = vectorizer.get_feature_names_out()
                
                # Get mean TF-IDF scores
                scores = X.mean(axis=0).A1
                keyword_scores = list(zip(feature_names, scores))
                keyword_scores.sort(key=lambda x: x[1], reverse=True)
                
                return keyword_scores[:top_n]
            except ValueError:
                # Fallback if vectorizer fails (e.g., too few documents)
                return self._fallback_keyword_extraction(text, top_n)
                
        except Exception as e:
            logger.error(f"Error extracting keywords: {e}")
            return []
    
    def _fallback_keyword_extraction(self, text: str, top_n: int) -> List[Tuple[str, float]]:
        """Fallback keyword extraction using frequency"""
        words = word_tokenize(text.lower())
        words = [w for w in words if w.isalnum() and w not in self.stop_words and len(w) > 3]
        
        freq_dist = FreqDist(words)
        keywords = [(word, freq / len(words)) for word, freq in freq_dist.most_common(top_n)]
        
        return keywords
    
    def summarize_text(self, text: str, num_sentences: int = 3) -> str:
        """Generate extractive summary using sentence scoring"""
        try:
            sentences = self.extract_sentences(text)
            
            if len(sentences) <= num_sentences:
                return text
            
            # Score sentences based on keyword frequency
            words = word_tokenize(text.lower())
            words = [w for w in words if w.isalnum() and w not in self.stop_words]
            freq_dist = FreqDist(words)
            
            sentence_scores = {}
            for i, sentence in enumerate(sentences):
                words_in_sentence = [w.lower() for w in word_tokenize(sentence) if w.isalnum()]
                score = sum(freq_dist[word] for word in words_in_sentence)
                sentence_scores[i] = score
            
            # Get top sentences in original order
            top_sentences_idx = sorted(
                sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:num_sentences]
            )
            
            summary = ' '.join(sentences[i] for i in top_sentences_idx)
            return summary
            
        except Exception as e:
            logger.error(f"Error summarizing text: {e}")
            return text[:500] + "..."
    
    def extract_important_points(self, text: str, num_points: int = 5) -> List[str]:
        """Extract important points from text"""
        try:
            sentences = self.extract_sentences(text)
            keywords = self.extract_keywords(text, top_n=20)
            
            if not sentences or not keywords:
                return [s[:100] + "..." for s in sentences[:num_points]]
            
            keyword_set = {kw[0].lower() for kw in keywords}
            
            # Score sentences by keyword presence
            sentence_scores = {}
            for i, sentence in enumerate(sentences):
                sentence_lower = sentence.lower()
                score = sum(1 for kw in keyword_set if kw in sentence_lower)
                sentence_scores[i] = score
            
            # Get top sentences
            top_sentences_idx = sorted(
                sentence_scores, 
                key=sentence_scores.get, 
                reverse=True
            )[:num_points]
            
            important_points = [sentences[i].strip() for i in sorted(top_sentences_idx)]
            return important_points
            
        except Exception as e:
            logger.error(f"Error extracting important points: {e}")
            return []
    
    def answer_question(self, text: str, question: str) -> Dict:
        """Find best answer to question from text"""
        try:
            sentences = self.extract_sentences(text)
            
            if not sentences:
                return {
                    "answer": "No relevant content found in the text.",
                    "confidence": 0,
                    "source_sentence": ""
                }
            
            # Tokenize question
            question_words = set(w.lower() for w in word_tokenize(question) 
                               if w.isalnum() and w not in self.stop_words)
            
            if not question_words:
                return {
                    "answer": "Could not process question. Please try a different question.",
                    "confidence": 0,
                    "source_sentence": ""
                }
            
            # Score sentences based on word overlap
            sentence_scores = {}
            for i, sentence in enumerate(sentences):
                sentence_words = set(w.lower() for w in word_tokenize(sentence) if w.isalnum())
                overlap = len(question_words & sentence_words)
                sentence_scores[i] = overlap
            
            # Get best matching sentence
            if max(sentence_scores.values()) == 0:
                return {
                    "answer": "No relevant answer found in the text for this question.",
                    "confidence": 0,
                    "source_sentence": ""
                }
            
            best_idx = max(sentence_scores, key=sentence_scores.get)
            best_sentence = sentences[best_idx]
            confidence = sentence_scores[best_idx] / len(question_words)
            
            return {
                "answer": best_sentence,
                "confidence": min(confidence, 1.0),
                "source_sentence": best_sentence
            }
            
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            return {
                "answer": f"Error processing question: {str(e)}",
                "confidence": 0,
                "source_sentence": ""
            }


# Create singleton instance
text_analyzer = TextAnalyzer()
