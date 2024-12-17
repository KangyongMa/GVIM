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
from matplotlib import font_manager
import matplotlib as mpl

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

title_font = {'family': 'Times New Roman',
              'weight': 'bold',
              'size': 14}

label_font = {'family': 'Times New Roman',
              'weight': 'normal',
              'size': 12}

tick_font = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 10}

legend_font = {'family': 'Times New Roman',
               'weight': 'normal',
               'size': 10}

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

            # Add sophisticated unit conversion 
            if response_unit != correct_unit:
                response_value = self.convert_to_standard_unit(response_value, response_unit, correct_unit)

            # Handle special cases for very small values
            if abs(correct_value) < 1e-10:
                return 1 if abs(response_value) < 1e-10 else 0.1

            # Calculate relative error with weighted bands
            relative_error = abs(response_value - correct_value) / abs(correct_value)

            if relative_error == 0:
                return 1.0  # Perfect match
            elif relative_error < 0.01:
                return 0.9  # Within 1% error 
            elif relative_error < 0.1:
                return 0.7  # Within 10% error
            elif relative_error < 0.5:
                return 0.5  # Within 50% error
            elif relative_error < 1:
                return 0.3  # Within 100% error
            else:
                return max(0.1, 1 - min(1, np.log10(relative_error + 1) / 2))

        except Exception as e:
            logging.error(f"Error in calculate_numeric_accuracy: {e}")
            return 0.1

    def calculate_keyword_score(self, text, question):
        score = 0
        text_lower = text.lower()
        question_lower = question.lower()
        
        # Weight different types of chemical terms
        weights = {
            'reaction_terms': 2.0,
            'structure_terms': 1.5,
            'property_terms': 1.0,
            'general_terms': 0.5
        }
        
        # Define term categories
        chemical_terms = {
            'reaction_terms': ['catalyst', 'synthesis', 'yield', 'mechanism'],
            'structure_terms': ['bond', 'orbital', 'molecule', 'atom'],
            'property_terms': ['energy', 'temperature', 'pressure', 'concentration'],
            'general_terms': ['compound', 'solution', 'mixture']
        }
        
        # Score each category
        for category, terms in chemical_terms.items():
            category_score = 0
            for term in terms:
                if term in text_lower:
                    category_score += 1
            score += (category_score / len(terms)) * weights[category]
            
        # Context matching bonus
        context_bonus = 0
        for term in chemical_terms['reaction_terms']:
            if term in question_lower and term in text_lower:
                context_bonus += 0.5
                
        score = (score + context_bonus) / sum(weights.values())
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
        # Base similarity score
        similarity = self.calculate_similarity(response, correct_answer)
        
        # Chemical property specific checks
        if any(term in question.lower() for term in ['lumo', 'homo', 'orbital', 'energy']):
            response_value, _ = self.extract_numeric_value_and_unit(response)
            correct_value, _ = self.extract_numeric_value_and_unit(correct_answer)
            if response_value is not None and correct_value is not None:
                # Reward correct sign of energy values
                if (response_value < 0) == (correct_value < 0):
                    similarity += 0.2
        
        # MOF structure specific checks
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

    def plot_performance_distribution(self, results, ax, colors):
        """Plot performance distribution using violin plots"""
        model_scores = []
        labels = []
        
        for model in self.models:
            scores = [score['final_score'] for result in results['individual_results']
                     for model_name, score in result['model_scores'].items() if model_name == model]
            model_scores.append(scores)
            labels.append(model)
        
        parts = ax.violinplot(model_scores, points=100, showmeans=True)
        
        # Customize violin plots
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i % len(colors)])
            pc.set_alpha(0.7)
        
        parts['cmeans'].set_color('black')
        parts['cmeans'].set_linewidth(1.5)
        
        ax.set_xticks(range(1, len(labels) + 1))
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        
        ax.set_ylabel('Score Distribution', fontfamily='Times New Roman', fontsize=12)
        ax.set_title('(a) Model Performance Distribution', fontfamily='Times New Roman', 
                     fontweight='bold', fontsize=14, pad=20)
        
        # Add mean values as text
        for i, scores in enumerate(model_scores, 1):
            mean = np.mean(scores)
            ax.text(i, ax.get_ylim()[1], f'{mean:.2f}',
                    ha='center', va='bottom', fontfamily='Times New Roman', fontsize=10)

    def plot_enhanced_type_performance(self, results, ax, colors):
        """Plot enhanced type performance using grouped box plots"""
        positions = []
        data = []
        labels = []
        
        for i, qtype in enumerate(self.question_types):
            for j, model in enumerate(self.models):
                scores = [result['model_scores'][model]['final_score']
                         for result in results['individual_results']
                         if result['question_type'] == qtype]
                pos = i + j/(len(self.models) + 1)
                positions.append(pos)
                data.append(scores)
                labels.append(model if i == 0 else '')
        
        bplot = ax.boxplot(data, positions=positions, patch_artist=True,
                          widths=0.1, medianprops=dict(color="black"))
        
        for i, patch in enumerate(bplot['boxes']):
            patch.set_facecolor(colors[i % len(self.models)])
            patch.set_alpha(0.7)
        
        ax.set_xticks([i + 0.5 for i in range(len(self.question_types))])
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.set_xticklabels(self.question_types, fontfamily='Times New Roman', fontsize=10)
        ax.set_ylabel('Performance Score', fontfamily='Times New Roman', fontsize=12)
        ax.set_title('(b) Performance by Question Type', fontfamily='Times New Roman',
                     fontweight='bold', fontsize=14, pad=20)
        
        handles = [plt.Rectangle((0,0),1,1, facecolor=colors[i], alpha=0.7)
                   for i in range(len(self.models))]
        ax.legend(handles, self.models, loc='upper right', bbox_to_anchor=(1.15, 1),
                 prop={'family': 'Times New Roman', 'size': 10})

    def plot_criteria_heatmap(self, results, ax):
        """Plot criteria correlation heatmap"""
        criteria = list(results['model_criteria_scores'][self.models[0]].keys())
        correlation_matrix = np.zeros((len(criteria), len(criteria)))
        
        for i, c1 in enumerate(criteria):
            for j, c2 in enumerate(criteria):
                scores1 = []
                scores2 = []
                for model in self.models:
                    for result in results['individual_results']:
                        if c1 in result['model_scores'][model]['criteria_scores'] and \
                           c2 in result['model_scores'][model]['criteria_scores']:
                            scores1.append(result['model_scores'][model]['criteria_scores'][c1])
                            scores2.append(result['model_scores'][model]['criteria_scores'][c2])
                if scores1 and scores2:
                    correlation_matrix[i, j] = np.corrcoef(scores1, scores2)[0, 1]
        
        sns.heatmap(correlation_matrix, annot=True, cmap='RdBu_r', center=0,
                    xticklabels=criteria, yticklabels=criteria, ax=ax,
                    annot_kws={'size': 8})
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.set_xticklabels(criteria, rotation=45, ha='right')
        ax.set_yticklabels(criteria, rotation=0)
        ax.set_title('(c) Criteria Correlation Matrix', fontfamily='Times New Roman',
                     fontweight='bold', fontsize=14, pad=20)

    def plot_criteria_radar(self, results, ax, colors):
        """Plot radar chart for criteria comparison"""
        criteria = list(results['model_criteria_scores'][self.models[0]].keys())
        angles = np.linspace(0, 2*np.pi, len(criteria), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))
        
        for i, model in enumerate(self.models):
            values = [results['model_criteria_scores'][model][criterion] for criterion in criteria]
            values = np.concatenate((values, [values[0]]))
            
            ax.plot(angles, values, 'o-', color=colors[i % len(colors)],
                    label=model, alpha=0.7, linewidth=2)
            ax.fill(angles, values, color=colors[i % len(colors)], alpha=0.1)
        
        ax.set_xticks(angles[:-1])
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.set_xticklabels(criteria, fontfamily='Times New Roman', fontsize=10)
        ax.set_title('(d) Criteria Performance Comparison', fontfamily='Times New Roman',
                     fontweight='bold', fontsize=14, pad=20)
        
        ax.legend(loc='center left', bbox_to_anchor=(1.2, 0.5),
                 prop={'family': 'Times New Roman', 'size': 10})

    def plot_results(self, results):
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.serif'] = ['Times New Roman']
        plt.rcParams['axes.unicode_minus'] = False
        
        nature_colors = ['#4878D0', '#EE854A', '#6ACC64', '#D65F5F', '#956CB4', 
                        '#8C613C', '#DC7EC0', '#797979']
        
        fig = plt.figure(figsize=(20, 15))
        gs = plt.GridSpec(2, 2, figure=fig)
        
        ax1 = fig.add_subplot(gs[0, 0])
        self.plot_performance_distribution(results, ax1, nature_colors)
        
        ax2 = fig.add_subplot(gs[0, 1])
        self.plot_enhanced_type_performance(results, ax2, nature_colors)
        
        ax3 = fig.add_subplot(gs[1, 0])
        self.plot_criteria_heatmap(results, ax3)
        
        ax4 = fig.add_subplot(gs[1, 1], projection='polar')
        self.plot_criteria_radar(results, ax4, nature_colors)
        
        plt.tight_layout()
        plt.savefig('combined_analysis.png', dpi=600, bbox_inches='tight')
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
    
    questions_file = 'yourpath//txt 2.json'
    questions_data = load_questions(questions_file)
    
    if questions_data:
        results = evaluator.score_multiple(questions_data)
        evaluator.plot_results(results)
        
        with open('evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        logging.info("Evaluation completed. Results saved to evaluation_results.json")
        logging.info("Performance plots saved as combined_analysis")
    else:
        logging.error("Failed to load questions. Exiting.")

if __name__ == "__main__":
    main()
