import json
import logging
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from rouge import Rouge
import textstat

try:
    import Levenshtein
except ImportError:
    print("Levenshtein module not found. Using a simple similarity measure instead.")
    Levenshtein = None

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class OptimizedModelEvaluator:
    def __init__(self, config):
        self.config = config
        self.models = config.get('models', [])
        self.question_types = config.get('question_types', [])
        self.scoring_criteria = config.get('scoring_criteria', {})
        self.keyword_importance = config.get('keyword_importance', {})
        self.chemical_terms = config.get('chemical_terms', [])
        self.unit_conversions = config.get('unit_conversions', {})
        nltk.download('punkt', quiet=True)
        self.rouge = Rouge()
        logging.basicConfig(level=logging.DEBUG)

    def is_numeric(self, text):
        text = str(text).strip()
        if text == "0" or text == "0.0":
            return True
        try:
            float(text.replace(',', ''))
            return True
        except ValueError:
            return bool(re.match(r'^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?\s*[a-zA-Z/]*$', text))

    def extract_numeric_value_and_unit(self, text):
        match = re.search(r'(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*([a-zA-Z/]*)', text)
        if match:
            value = float(match.group(1))
            unit = match.group(2).lower() if match.group(2) else ''
            return value, unit
        return None, None

    def convert_to_standard_unit(self, value, unit, standard_unit):
        if unit == standard_unit:
            return value
        conversion_key = f"{unit}_to_{standard_unit}"
        if conversion_key in self.unit_conversions:
            return value * self.unit_conversions[conversion_key]
        return value

    def calculate_numeric_accuracy(self, response, correct_answer):
        try:
            response_value, response_unit = self.extract_numeric_value_and_unit(response)
            correct_value, correct_unit = self.extract_numeric_value_and_unit(correct_answer)

            if response_value is None or correct_value is None:
                return 0.1

            if response_unit != correct_unit:
                response_value = self.convert_to_standard_unit(response_value, response_unit, correct_unit)

            if abs(correct_value) < 1e-10:
                return 1 if abs(response_value) < 1e-10 else 0.1

            relative_error = abs(response_value - correct_value) / abs(correct_value)
            
            if relative_error == 0:
                return 1.0
            elif relative_error < 0.01:
                return 0.9
            elif relative_error < 0.1:
                return 0.7
            elif relative_error < 0.5:
                return 0.5
            elif relative_error < 1:
                return 0.3
            else:
                return max(0.1, 1 - min(1, np.log10(relative_error + 1) / 2))

        except Exception as e:
            logging.error(f"Error in calculate_numeric_accuracy: {e}")
            return 0.1

    def calculate_keyword_score(self, text, question):
        score = 0
        text_lower = text.lower()
        question_lower = question.lower()
        
        for keyword, importance in self.keyword_importance.items():
            if keyword.lower() in text_lower:
                score += importance
        
        for term in self.chemical_terms:
            if term in question_lower and term in text_lower:
                score += 0.5
        
        return min(score, 1)

    def calculate_similarity(self, answer, correct_answer):
        if Levenshtein:
            return 1 - (Levenshtein.distance(answer.lower(), correct_answer.lower()) / max(len(answer), len(correct_answer)))
        else:
            answer_words = set(answer.lower().split())
            correct_words = set(correct_answer.lower().split())
            return len(answer_words.intersection(correct_words)) / max(len(answer_words), len(correct_words))

    def calculate_bleu_score(self, response, correct_answer):
        reference = [word_tokenize(correct_answer.lower())]
        candidate = word_tokenize(response.lower())
        smoothie = SmoothingFunction().method1
        return sentence_bleu(reference, candidate, smoothing_function=smoothie)

    def calculate_rouge_scores(self, response, correct_answer):
        scores = self.rouge.get_scores(response, correct_answer)[0]
        return {
            'rouge-1': scores['rouge-1']['f'],
            'rouge-2': scores['rouge-2']['f'],
            'rouge-l': scores['rouge-l']['f']
        }

    def calculate_readability(self, text):
        score = textstat.flesch_reading_ease(text)
        return max(0, min(score, 100)) / 100

    def calculate_coherence(self, text):
        sentences = nltk.sent_tokenize(text)
        if len(sentences) < 2:
            return 1.0

        coherence_scores = []
        for i in range(len(sentences) - 1):
            words1 = set(word_tokenize(sentences[i].lower()))
            words2 = set(word_tokenize(sentences[i + 1].lower()))
            overlap = len(words1.intersection(words2))
            coherence_scores.append(overlap / max(len(words1), len(words2)))

        return sum(coherence_scores) / len(coherence_scores)

    def calculate_conciseness(self, response, correct_answer):
        response_value, response_unit = self.extract_numeric_value_and_unit(response)
        correct_value, correct_unit = self.extract_numeric_value_and_unit(correct_answer)
        
        if response_value is None or correct_value is None:
            return 0.5
        
        response_precision = len(str(response_value).split('.')[-1])
        correct_precision = len(str(correct_value).split('.')[-1])
        
        precision_score = 1 - min(abs(response_precision - correct_precision) / 10, 1)
        length_score = 1 - min(abs(len(response) - len(correct_answer)) / len(correct_answer), 1)
        
        return (precision_score + length_score) / 2

    def calculate_factual_accuracy(self, response, correct_answer, question):
        similarity = self.calculate_similarity(response, correct_answer)
        
        if any(term in question.lower() for term in ['lumo', 'homo', 'orbital', 'energy']):
            response_value, _ = self.extract_numeric_value_and_unit(response)
            correct_value, _ = self.extract_numeric_value_and_unit(correct_answer)
            if response_value is not None and correct_value is not None:
                if (response_value < 0) == (correct_value < 0):
                    similarity += 0.2
        
        elif 'mof' in question.lower():
            important_parts = ['linker', 'node', 'topology']
            for part in important_parts:
                if part in response.lower() and part in correct_answer.lower():
                    similarity += 0.1
        
        return min(similarity, 1)

    def calculate_creativity(self, response, correct_answer):
        uniqueness = 1 - self.calculate_similarity(response, correct_answer)
        coherence = self.calculate_coherence(response)
        return (uniqueness + coherence) / 2

    def score(self, question, responses, correct_answer, provided_type):
        question_type = provided_type  # Use the provided type directly
        keyword_score = self.calculate_keyword_score(question, question)

        scores = {}
        for model, response in responses.items():
            criteria_scores = {}
            for criterion, weight in self.scoring_criteria.get(question_type, {}).items():
                if criterion == 'numeric_accuracy' and question_type == 'numeric':
                    criteria_scores[criterion] = self.calculate_numeric_accuracy(response, correct_answer)
                elif criterion == 'similarity':
                    criteria_scores[criterion] = self.calculate_similarity(response, correct_answer)
                elif criterion == 'keyword_relevance':
                    criteria_scores[criterion] = self.calculate_keyword_score(response, question)
                elif criterion == 'bleu_score':
                    criteria_scores[criterion] = self.calculate_bleu_score(response, correct_answer)
                elif criterion == 'rouge_scores':
                    rouge_scores = self.calculate_rouge_scores(response, correct_answer)
                    criteria_scores[criterion] = sum(rouge_scores.values()) / len(rouge_scores)
                elif criterion == 'readability':
                    criteria_scores[criterion] = self.calculate_readability(response)
                elif criterion == 'coherence':
                    criteria_scores[criterion] = self.calculate_coherence(response)
                elif criterion == 'conciseness':
                    criteria_scores[criterion] = self.calculate_conciseness(response, correct_answer)
                elif criterion == 'factual_accuracy':
                    criteria_scores[criterion] = self.calculate_factual_accuracy(response, correct_answer, question)
                elif criterion == 'creativity' and question_type == 'generate':
                    criteria_scores[criterion] = self.calculate_creativity(response, correct_answer)

            final_score = sum(criteria_scores[c] * w for c, w in self.scoring_criteria.get(question_type, {}).items())
            final_score = min(max(final_score * 10, 0), 10)  # Scale to 0-10 and clamp

            scores[model] = {
                'criteria_scores': criteria_scores,
                'final_score': final_score
            }

        return {
            'question_type': question_type,
            'keyword_score': keyword_score,
            'model_scores': scores
        }

    def score_multiple(self, questions_data):
        model_total_scores = {model: 0 for model in self.models}
        model_type_scores = {model: {qtype: [] for qtype in self.question_types} for model in self.models}
        model_criteria_scores = {model: {} for model in self.models}

        results = []
        for i, question_data in enumerate(questions_data, 1):
            question = question_data['question']
            correct_answer = question_data['correct_answer']
            responses = question_data['responses']
            provided_type = question_data.get('type', '')  # Get the provided type from the JSON

            logging.info(f"Question {i}: Type '{provided_type}', Q: '{question}', A: '{correct_answer}'")

            result = self.score(question, responses, correct_answer, provided_type)
            results.append(result)

            for model, score in result['model_scores'].items():
                model_total_scores[model] += score['final_score']
                model_type_scores[model][result['question_type']].append(score['final_score'])
                for criterion, criterion_score in score['criteria_scores'].items():
                    if criterion not in model_criteria_scores[model]:
                        model_criteria_scores[model][criterion] = []
                    model_criteria_scores[model][criterion].append(criterion_score)

            if i % 10 == 0:
                logging.info(f"Processed {i} questions")

        num_questions = len(questions_data)
        model_average_scores = {model: total / num_questions for model, total in model_total_scores.items()}

        for model in self.models:
            for qtype in self.question_types:
                if model_type_scores[model][qtype]:
                    model_type_scores[model][qtype] = sum(model_type_scores[model][qtype]) / len(model_type_scores[model][qtype])
                else:
                    model_type_scores[model][qtype] = 0

            for criterion in model_criteria_scores[model]:
                if model_criteria_scores[model][criterion]:
                    model_criteria_scores[model][criterion] = sum(model_criteria_scores[model][criterion]) / len(model_criteria_scores[model][criterion])
                else:
                    model_criteria_scores[model][criterion] = 0

        return {
            'individual_results': results,
            'model_total_scores': model_total_scores,
            'model_average_scores': model_average_scores,
            'model_type_scores': model_type_scores,
            'model_criteria_scores': model_criteria_scores
        }

    def calculate_error(self, data):
        """
        Calculate standard error, limiting the maximum error value
        """
        n = len(data)
        if n == 0:
            return 0
        se = np.std(data, ddof=1) / np.sqrt(n)  # Use standard error
        return min(se, 0.5)  # Limit maximum error value to 0.5

    def plot_results(self, results):
        plt.style.use('seaborn-v0_8-whitegrid')
        
        palette1 = ['#92A5D1', '#C5DFF4', '#AEB2D1', '#D9B9D4', '#E8D0E8', '#F0E5E5', '#FFD700', '#FF69B4']  # Added two more colors
        palette2 = ['#C25759', '#E69191', '#EDB8B0', '#F5DFDB', '#F8ECEC', '#FAF5F5', '#FFA07A', '#FF4500']  # Added two more colors
        palette3 = ['#7C9B95', '#C9DCC4', '#DAA87C', '#F4EEAC', '#F9F5D7', '#FCF9F0', '#98FB98', '#00CED1']  # Added two more colors
        
        # Create 8-color palettes for each plot
        overall_palette = palette1
        type_palette = palette2
        total_palette = palette3
        criteria_palette = palette1
        
        self.plot_overall_performance(results, overall_palette)
        self.plot_performance_by_type(results, type_palette)
        self.plot_total_scores(results, total_palette)
        self.plot_criteria_scores(results, criteria_palette)

    def plot_overall_performance(self, results, palette):
        fig, ax = plt.subplots(figsize=(14, 7))
        models = list(results['model_average_scores'].keys())
        scores = list(results['model_average_scores'].values())
        
        errors = []
        for model in models:
            model_scores = [score['final_score'] for result in results['individual_results'] 
                            for model_name, score in result['model_scores'].items() if model_name == model]
            errors.append(self.calculate_error(model_scores))
        
        bars = ax.bar(models, scores, color=palette[:len(models)], alpha=0.8)
        
        ax.errorbar(models, scores, yerr=errors, fmt='none', ecolor='black', capsize=3)
        
        ax.set_title('Overall Model Performance', fontsize=16, fontweight='bold')
        ax.set_xlabel('Models', fontsize=12)
        ax.set_ylabel('Average Score', fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=10)
        
        y_max = max(scores) + max(errors) + 0.5
        ax.set_ylim(0, y_max)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(errors) + 0.1,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig('overall_performance.png', dpi=600, bbox_inches='tight')
        plt.close()

    def plot_performance_by_type(self, results, palette):
        fig, ax = plt.subplots(figsize=(18, 8))  # Increased figure width
        x = np.arange(len(self.question_types))
        width = 0.10  # Adjusted for 8 models
        
        for i, model in enumerate(self.models):
            scores = [results['model_type_scores'][model][qtype] for qtype in self.question_types]
            errors = [self.calculate_error([result['model_scores'][model]['final_score'] 
                                            for result in results['individual_results'] 
                                            if result['question_type'] == qtype]) 
                      for qtype in self.question_types]
            
            bars = ax.bar(x + i*width, scores, width, label=model, color=palette[i % len(palette)], alpha=0.8)
            
            ax.errorbar(x + i*width, scores, yerr=errors, fmt='none', ecolor='black', capsize=3)
            
            for j, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + errors[j] + 0.1,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=8, rotation=0)
        
        ax.set_title('Model Performance by Question Type', fontsize=16, fontweight='bold')
        ax.set_xlabel('Question Types', fontsize=12)
        ax.set_ylabel('Average Score', fontsize=12)
        ax.set_xticks(x + width * (len(self.models) - 1) / 2)
        ax.set_xticklabels(self.question_types, rotation=0, ha='center')
        ax.legend(title='Models', title_fontsize='12', fontsize='10', loc='upper left', bbox_to_anchor=(1, 1))
        ax.tick_params(axis='both', which='major', labelsize=10)
        
        y_max = max([max(results['model_type_scores'][model].values()) for model in self.models]) + 0.5
        ax.set_ylim(0, y_max)
        
        plt.tight_layout()
        plt.savefig('performance_by_type.png', dpi=600, bbox_inches='tight')
        plt.close()

    def plot_total_scores(self, results, palette):
        fig, ax = plt.subplots(figsize=(14, 7))
        models = list(results['model_total_scores'].keys())
        scores = list(results['model_total_scores'].values())
        
        errors = []
        for model in models:
            model_scores = [score['final_score'] for result in results['individual_results'] 
                            for model_name, score in result['model_scores'].items() if model_name == model]
            errors.append(self.calculate_error(model_scores) * len(results['individual_results']))
        
        bars = ax.bar(models, scores, color=palette[:len(models)], alpha=0.8)
        
        ax.errorbar(models, scores, yerr=errors, fmt='none', ecolor='black', capsize=3)
        
        ax.set_title('Total Model Scores', fontsize=16, fontweight='bold')
        ax.set_xlabel('Models', fontsize=12)
        ax.set_ylabel('Total Score', fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=10)
        
        y_max = max(scores) + max(errors) + 20
        ax.set_ylim(0, y_max)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(errors) + 5,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig('total_scores.png', dpi=600, bbox_inches='tight')
        plt.close()


    def plot_criteria_scores(self, results, palette):
        fig, ax = plt.subplots(figsize=(18, 8))  # Increased figure width
        criteria = list(results['model_criteria_scores'][self.models[0]].keys())
        x = np.arange(len(criteria))
        width = 0.10  # Adjusted for 8 models

        for i, model in enumerate(self.models):
            scores = [results['model_criteria_scores'][model][criterion] for criterion in criteria]
            errors = [self.calculate_error([result['model_scores'][model]['criteria_scores'].get(criterion, 0) 
                                            for result in results['individual_results']]) 
                      for criterion in criteria]
            
            bars = ax.bar(x + i*width, scores, width, label=model, color=palette[i % len(palette)], alpha=0.8)
            
            ax.errorbar(x + i*width, scores, yerr=errors, fmt='none', ecolor='black', capsize=3)
            
            for j, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + errors[j] + 0.01,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=8, rotation=0)
        
        ax.set_title('Model Performance by Criteria', fontsize=16, fontweight='bold')
        ax.set_xlabel('Criteria', fontsize=12)
        ax.set_ylabel('Average Score', fontsize=12)
        ax.set_xticks(x + width * (len(self.models) - 1) / 2)
        ax.set_xticklabels(criteria, rotation=45, ha='right')
        ax.legend(title='Models', title_fontsize='12', fontsize='10', loc='upper left', bbox_to_anchor=(1, 1))
        ax.tick_params(axis='both', which='major', labelsize=10)
        
        y_min = min([min(results['model_criteria_scores'][model].values()) for model in self.models])
        y_max = max([max(results['model_criteria_scores'][model].values()) for model in self.models])
        
        y_padding = (y_max - y_min) * 0.1
        y_min -= y_padding
        y_max += y_padding
        
        if y_min > 0:
            y_min = 0
        elif y_max < 0:
            y_max = 0
        
        ax.set_ylim(y_min, y_max)
        
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        plt.savefig('criteria_scores.png', dpi=600, bbox_inches='tight')
        plt.close()

def load_questions(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            try:
                questions_data = json.loads(content)
            except json.JSONDecodeError as e:
                logging.error(f"JSON decode error: {str(e)}")
                lines = content.split('\n')
                error_line = lines[e.lineno - 1]
                logging.error(f"Error on line {e.lineno}, column {e.colno}:")
                logging.error(error_line)
                logging.error(" " * (e.colno - 1) + "^")
                return None
        logging.info(f"Successfully loaded {len(questions_data)} questions from {file_path}")
        return questions_data
    except Exception as e:
        logging.error(f"Error loading questions from {file_path}: {str(e)}")
        return None

def main():
    config = {
        'models': ["Llama3", "Mistral", "Phi-3", "Gemma", "Gemma2", "Phi-3 Medium", "MistralNemo", "Llama3.1"],
        'question_types': ['numeric', 'descriptive', 'generate'],
        'scoring_criteria': {
            'numeric': {
                'numeric_accuracy': 0.6,
                'keyword_relevance': 0.2,
                'conciseness': 0.2
            },
            'descriptive': {
                'bleu_score': 0.2,
                'rouge_scores': 0.2,
                'keyword_relevance': 0.2,
                'readability': 0.2,
                'coherence': 0.2
            },
            'generate': {
                'creativity': 0.4,
                'coherence': 0.3,
                'keyword_relevance': 0.3
            }
        },
        'keyword_importance': {
            'reaction': 0.5,
            'mechanism': 0.5,
            'synthesis': 0.5,
            'catalyst': 0.5,
            'bond': 0.3,
            'electron': 0.3,
            'orbital': 0.3
        },
        'chemical_terms': ['alkane', 'alkene', 'alkyne', 'aromatic', 'nucleophile', 'electrophile'],
        'unit_conversions': {
            'kJ_to_kcal': 0.239006,
            'kcal_to_kJ': 4.184,
            'eV_to_kJ': 96.485,
            'kJ_to_eV': 0.0103643
        }
    }
    evaluator = OptimizedModelEvaluator(config)
    
    questions_file = 'E://HuaweiMoveData//Users//makangyong//Desktop//txt 2.json'
    questions_data = load_questions(questions_file)
    
    if questions_data:
        results = evaluator.score_multiple(questions_data)
        evaluator.plot_results(results)
        
        with open('evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        logging.info("Evaluation completed. Results saved to evaluation_results.json")
        logging.info("Performance plots saved as overall_performance.png, performance_by_type.png, total_scores.png, and criteria_scores.png")
    else:
        logging.error("Failed to load questions. Exiting.")

if __name__ == "__main__":
    main()