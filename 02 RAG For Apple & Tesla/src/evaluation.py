"""
Evaluation script for RAG system.
Includes metrics for accuracy, hallucination detection, and answer quality.
"""
import json
from pathlib import Path
from typing import List, Dict, Any
import re
import numpy as np
from loguru import logger

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    logger.warning("rouge-score not available")


class RAGEvaluator:
    """Evaluator for RAG system performance."""
    
    def __init__(self):
        """Initialize evaluator."""
        self.rouge_scorer = None
        if ROUGE_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(
                ['rouge1', 'rouge2', 'rougeL'],
                use_stemmer=True
            )
        
        # Ground truth answers
        self.ground_truth = {
            1: {"answer": "$391,036 million", "source": "Apple 10-K, Item 8, p. 282"},
            2: {"answer": "15,115,823,000 shares", "source": "Apple 10-K, first paragraph"},
            3: {"answer": "$96,662 million", "source": "Apple 10-K, Item 8, Note 9, p. 394"},
            4: {"answer": "November 1, 2024", "source": "Apple 10-K, Signature page"},
            5: {"answer": "No. Checkmark indicates 'No' under Item 1B", "source": "Apple 10-K, Item 1B, p. 176"},
            6: {"answer": "$96,773 million", "source": "Tesla 10-K, Item 7"},
            7: {"answer": "~84% ($81,924M / $96,773M)", "source": "Tesla 10-K, Item 7"},
            8: {"answer": "Central to strategy, innovation, leadership; loss could disrupt", "source": "Tesla 10-K, Item 1A"},
            9: {"answer": "Model S, Model 3, Model X, Model Y, Cybertruck", "source": "Tesla 10-K, Item 1"},
            10: {"answer": "Finance solar systems with investors; customers sign PPAs", "source": "Tesla 10-K, Item 7"},
            11: {"answer": "Not answerable", "source": "N/A"},
            12: {"answer": "Not answerable", "source": "N/A"},
            13: {"answer": "Not answerable", "source": "N/A"}
        }
    
    def evaluate(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate RAG results against ground truth.
        
        Args:
            results: List of RAG responses
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Evaluating RAG results...")
        
        metrics = {
            'total_questions': len(results),
            'per_question': [],
            'aggregate': {}
        }
        
        for result in results:
            q_id = result['question_id']
            answer = result['answer']
            sources = result.get('sources', [])
            
            # Evaluate this question
            q_metrics = self._evaluate_question(q_id, answer, sources)
            metrics['per_question'].append(q_metrics)
        
        # Calculate aggregate metrics
        metrics['aggregate'] = self._calculate_aggregate_metrics(
            metrics['per_question']
        )
        
        return metrics
    
    def _evaluate_question(
        self,
        question_id: int,
        answer: str,
        sources: List[str]
    ) -> Dict[str, Any]:
        """Evaluate a single question."""
        gt = self.ground_truth.get(question_id, {})
        gt_answer = gt.get('answer', '')
        
        metrics = {
            'question_id': question_id,
            'answer': answer,
            'ground_truth': gt_answer,
            'exact_match': False,
            'contains_key_info': False,
            'has_hallucination': False,
            'hallucination_details': None,
            'proper_refusal': False,
            'has_sources': len(sources) > 0,
            'source_count': len(sources)
        }
        
        # Check for out-of-scope questions (11-13)
        if question_id in [11, 12, 13]:
            metrics['proper_refusal'] = self._check_refusal(answer)
            metrics['contains_key_info'] = metrics['proper_refusal']
        else:
            # Check exact match
            metrics['exact_match'] = self._check_exact_match(answer, gt_answer)
            
            # Check if answer contains key information
            metrics['contains_key_info'] = self._check_key_info(answer, gt_answer)
            
            # Detect hallucinations
            hallucination_result = self._detect_hallucination(
                answer, gt_answer, question_id
            )
            metrics['has_hallucination'] = hallucination_result['has_hallucination']
            metrics['hallucination_details'] = hallucination_result['details']
        
        # Calculate ROUGE scores if available
        if self.rouge_scorer and gt_answer:
            rouge_scores = self.rouge_scorer.score(gt_answer, answer)
            metrics['rouge1'] = rouge_scores['rouge1'].fmeasure
            metrics['rouge2'] = rouge_scores['rouge2'].fmeasure
            metrics['rougeL'] = rouge_scores['rougeL'].fmeasure
        
        return metrics
    
    def _check_exact_match(self, answer: str, ground_truth: str) -> bool:
        """Check if answer exactly matches ground truth."""
        # Normalize
        answer_clean = re.sub(r'[^\w\s]', '', answer.lower())
        gt_clean = re.sub(r'[^\w\s]', '', ground_truth.lower())
        
        return answer_clean == gt_clean
    
    def _check_key_info(self, answer: str, ground_truth: str) -> bool:
        """Check if answer contains key information from ground truth."""
        # Extract numbers and key words
        answer_lower = answer.lower()
        gt_lower = ground_truth.lower()
        
        # Extract numbers
        gt_numbers = set(re.findall(r'\d+[,\d]*\.?\d*', ground_truth))
        answer_numbers = set(re.findall(r'\d+[,\d]*\.?\d*', answer))
        
        # Check if major numbers are present
        if gt_numbers:
            number_match = len(gt_numbers & answer_numbers) / len(gt_numbers) >= 0.5
            if number_match:
                return True
        
        # Check key words (excluding common words)
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were'}
        gt_words = set(gt_lower.split()) - common_words
        answer_words = set(answer_lower.split()) - common_words
        
        if gt_words:
            word_match = len(gt_words & answer_words) / len(gt_words) >= 0.4
            return word_match
        
        return False
    
    def _check_refusal(self, answer: str) -> bool:
        """Check if answer properly refuses out-of-scope question."""
        refusal_phrases = [
            "not specified in the document",
            "cannot be answered based on the provided documents",
            "not found in the provided",
            "not mentioned in the document",
            "not available in the document"
        ]
        
        answer_lower = answer.lower()
        return any(phrase in answer_lower for phrase in refusal_phrases)
    
    def _detect_hallucination(
        self,
        answer: str,
        ground_truth: str,
        question_id: int
    ) -> Dict[str, Any]:
        """
        Detect potential hallucinations in the answer.
        
        Returns:
            Dictionary with has_hallucination flag and details
        """
        result = {
            'has_hallucination': False,
            'details': []
        }
        
        # 1. Check for contradictory numbers
        gt_numbers = set(re.findall(r'\d+[,\d]*\.?\d*', ground_truth))
        answer_numbers = set(re.findall(r'\d+[,\d]*\.?\d*', answer))
        
        # Remove formatting for comparison
        gt_nums_clean = {n.replace(',', '') for n in gt_numbers}
        answer_nums_clean = {n.replace(',', '') for n in answer_numbers}
        
        # Check if answer has numbers not in ground truth
        extra_numbers = answer_nums_clean - gt_nums_clean
        if extra_numbers and gt_numbers:
            # Allow small differences (e.g., rounding)
            significant_diff = False
            for ans_num in extra_numbers:
                try:
                    ans_val = float(ans_num)
                    # Check if it's significantly different from GT numbers
                    for gt_num in gt_nums_clean:
                        try:
                            gt_val = float(gt_num)
                            if abs(ans_val - gt_val) / max(gt_val, 1) > 0.1:  # >10% difference
                                significant_diff = True
                                break
                        except ValueError:
                            continue
                except ValueError:
                    continue
            
            if significant_diff:
                result['has_hallucination'] = True
                result['details'].append(
                    f"Answer contains numbers not in ground truth: {extra_numbers}"
                )
        
        # 2. Check for fabricated company names
        answer_lower = answer.lower()
        if question_id <= 5:  # Apple questions
            if 'tesla' in answer_lower and 'apple' not in answer_lower:
                result['has_hallucination'] = True
                result['details'].append("Answer mentions Tesla for an Apple question")
        elif question_id <= 10:  # Tesla questions
            if 'apple' in answer_lower and 'tesla' not in answer_lower:
                result['has_hallucination'] = True
                result['details'].append("Answer mentions Apple for a Tesla question")
        
        # 3. Check for vague or unsupported claims
        vague_phrases = [
            'approximately', 'around', 'roughly', 'about',
            'significant', 'substantial', 'considerable'
        ]
        
        if any(phrase in answer_lower for phrase in vague_phrases):
            # If ground truth has specific number but answer is vague
            if gt_numbers and not answer_numbers:
                result['has_hallucination'] = True
                result['details'].append("Answer is vague when specific number expected")
        
        return result
    
    def _calculate_aggregate_metrics(
        self,
        per_question: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate aggregate metrics across all questions."""
        total = len(per_question)
        
        if total == 0:
            return {}
        
        # In-scope questions (1-10)
        in_scope = [q for q in per_question if q['question_id'] <= 10]
        # Out-of-scope questions (11-13)
        out_of_scope = [q for q in per_question if q['question_id'] > 10]
        
        metrics = {
            'total_questions': total,
            'in_scope_questions': len(in_scope),
            'out_of_scope_questions': len(out_of_scope),
        }
        
        # In-scope metrics
        if in_scope:
            metrics['exact_match_rate'] = sum(
                q['exact_match'] for q in in_scope
            ) / len(in_scope)
            
            metrics['key_info_accuracy'] = sum(
                q['contains_key_info'] for q in in_scope
            ) / len(in_scope)
            
            metrics['hallucination_rate'] = sum(
                q['has_hallucination'] for q in in_scope
            ) / len(in_scope)
            
            metrics['source_citation_rate'] = sum(
                q['has_sources'] for q in in_scope
            ) / len(in_scope)
            
            # ROUGE scores
            if self.rouge_scorer:
                metrics['avg_rouge1'] = np.mean([
                    q.get('rouge1', 0) for q in in_scope
                ])
                metrics['avg_rouge2'] = np.mean([
                    q.get('rouge2', 0) for q in in_scope
                ])
                metrics['avg_rougeL'] = np.mean([
                    q.get('rougeL', 0) for q in in_scope
                ])
        
        # Out-of-scope metrics
        if out_of_scope:
            metrics['refusal_accuracy'] = sum(
                q['proper_refusal'] for q in out_of_scope
            ) / len(out_of_scope)
        
        # Overall score
        metrics['overall_score'] = (
            metrics.get('key_info_accuracy', 0) * 0.4 +
            (1 - metrics.get('hallucination_rate', 1)) * 0.3 +
            metrics.get('refusal_accuracy', 0) * 0.2 +
            metrics.get('source_citation_rate', 0) * 0.1
        )
        
        return metrics
    
    def print_report(self, metrics: Dict[str, Any]):
        """Print evaluation report."""
        logger.info("="*80)
        logger.info("EVALUATION REPORT")
        logger.info("="*80)
        
        agg = metrics['aggregate']
        
        logger.info(f"\nTotal Questions: {agg['total_questions']}")
        logger.info(f"In-Scope: {agg['in_scope_questions']}, Out-of-Scope: {agg['out_of_scope_questions']}")
        
        logger.info("\n--- In-Scope Performance ---")
        logger.info(f"Exact Match Rate: {agg.get('exact_match_rate', 0):.2%}")
        logger.info(f"Key Info Accuracy: {agg.get('key_info_accuracy', 0):.2%}")
        logger.info(f"Hallucination Rate: {agg.get('hallucination_rate', 0):.2%}")
        logger.info(f"Source Citation Rate: {agg.get('source_citation_rate', 0):.2%}")
        
        if self.rouge_scorer:
            logger.info(f"\nROUGE-1: {agg.get('avg_rouge1', 0):.3f}")
            logger.info(f"ROUGE-2: {agg.get('avg_rouge2', 0):.3f}")
            logger.info(f"ROUGE-L: {agg.get('avg_rougeL', 0):.3f}")
        
        logger.info("\n--- Out-of-Scope Performance ---")
        logger.info(f"Refusal Accuracy: {agg.get('refusal_accuracy', 0):.2%}")
        
        logger.info(f"\n--- Overall Score: {agg.get('overall_score', 0):.2%} ---")
        
        # Hallucination details
        hallucinations = [
            q for q in metrics['per_question']
            if q['has_hallucination']
        ]
        
        if hallucinations:
            logger.warning(f"\n⚠️  Found {len(hallucinations)} potential hallucinations:")
            for q in hallucinations:
                logger.warning(f"  Q{q['question_id']}: {q['hallucination_details']}")
        
        logger.info("="*80)


def evaluate_results(results_path: Path):
    """
    Evaluate RAG results from JSON file.
    
    Args:
        results_path: Path to results JSON file
    """
    with open(results_path) as f:
        results = json.load(f)
    
    evaluator = RAGEvaluator()
    metrics = evaluator.evaluate(results)
    
    # Print report
    evaluator.print_report(metrics)
    
    # Save detailed metrics
    metrics_path = results_path.parent / "evaluation_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Detailed metrics saved to {metrics_path}")


if __name__ == "__main__":
    # Example usage
    results_path = Path("outputs/rag_results.json")
    if results_path.exists():
        evaluate_results(results_path)
    else:
        logger.error(f"Results file not found: {results_path}")