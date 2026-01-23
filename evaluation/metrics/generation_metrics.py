"""
生成指标模块
集成常用的文本生成评估指标：ROUGE, BLEU, BERTScore 等
"""

from typing import List, Optional, Dict, Any
from evaluation.base import BaseMetric, EvaluationResult


class ROUGEMetric(BaseMetric):
    """ROUGE 指标（使用 rouge-score 库）"""
    
    def __init__(self, rouge_type: str = "rougeL"):
        """
        初始化 ROUGE 指标
        
        Args:
            rouge_type: ROUGE 类型 (rouge1, rouge2, rougeL)
        """
        super().__init__(name=f"ROUGE-{rouge_type}", rouge_type=rouge_type)
        self.rouge_type = rouge_type
        self._scorer = None
    
    def _get_scorer(self):
        """延迟加载 ROUGE scorer"""
        if self._scorer is None:
            try:
                from rouge_score import rouge_scorer
                self._scorer = rouge_scorer.RougeScorer([self.rouge_type], use_stemmer=True)
            except ImportError:
                raise ImportError(
                    "rouge-score 库未安装。请运行: pip install rouge-score"
                )
        return self._scorer
    
    def compute(
        self, 
        predictions: str, 
        references: str, 
        **kwargs
    ) -> EvaluationResult:
        """
        计算 ROUGE 分数
        
        Args:
            predictions: 生成的文本
            references: 参考文本
            
        Returns:
            EvaluationResult
        """
        scorer = self._get_scorer()
        scores = scorer.score(references, predictions)
        rouge_score = scores[self.rouge_type]
        
        return EvaluationResult(
            metric_name=self.name,
            value=rouge_score.fmeasure,
            metadata={
                "precision": rouge_score.precision,
                "recall": rouge_score.recall,
                "fmeasure": rouge_score.fmeasure
            }
        )


class BLEUMetric(BaseMetric):
    """BLEU 指标（使用 nltk 或 sacrebleu）"""
    
    def __init__(self, max_order: int = 4):
        super().__init__(name="BLEU", max_order=max_order)
        self.max_order = max_order
    
    def compute(
        self, 
        predictions: str, 
        references: str, 
        **kwargs
    ) -> EvaluationResult:
        """
        计算 BLEU 分数
        
        Args:
            predictions: 生成的文本
            references: 参考文本
            
        Returns:
            EvaluationResult
        """
        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            
            # 分词
            pred_tokens = predictions.split()
            ref_tokens = [references.split()]
            
            # 计算 BLEU（使用平滑函数避免 0 分）
            smoothing = SmoothingFunction()
            bleu = sentence_bleu(
                ref_tokens, 
                pred_tokens,
                smoothing_function=smoothing.method1
            )
            
            return EvaluationResult(
                metric_name=self.name,
                value=bleu,
                metadata={"max_order": self.max_order}
            )
        except ImportError:
            raise ImportError(
                "nltk 库未安装。请运行: pip install nltk"
            )


class BERTScoreMetric(BaseMetric):
    """BERTScore 指标（使用预训练模型计算语义相似度）"""
    
    def __init__(self, model_type: str = "microsoft/deberta-xlarge-mnli", lang: str = "en"):
        super().__init__(name="BERTScore", model_type=model_type, lang=lang)
        self.model_type = model_type
        self.lang = lang
    
    def compute(
        self, 
        predictions: str, 
        references: str, 
        **kwargs
    ) -> EvaluationResult:
        """
        计算 BERTScore
        
        Args:
            predictions: 生成的文本
            references: 参考文本
            
        Returns:
            EvaluationResult
        """
        try:
            from bert_score import score
            
            # 计算 BERTScore
            P, R, F1 = score(
                [predictions], 
                [references], 
                model_type=self.model_type,
                lang=self.lang,
                verbose=False
            )
            
            return EvaluationResult(
                metric_name=self.name,
                value=F1.item(),
                metadata={
                    "precision": P.item(),
                    "recall": R.item(),
                    "f1": F1.item(),
                    "model_type": self.model_type
                }
            )
        except ImportError:
            raise ImportError(
                "bert-score 库未安装。请运行: pip install bert-score"
            )


class SemanticSimilarity(BaseMetric):
    """语义相似度（使用 sentence-transformers）"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        super().__init__(name="SemanticSimilarity", model_name=model_name)
        self.model_name = model_name
        self._model = None
    
    def _get_model(self):
        """延迟加载模型"""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except ImportError:
                raise ImportError(
                    "sentence-transformers 库未安装。请运行: pip install sentence-transformers"
                )
        return self._model
    
    def compute(
        self, 
        predictions: str, 
        references: str, 
        **kwargs
    ) -> EvaluationResult:
        """
        计算语义相似度（余弦相似度）
        
        Args:
            predictions: 生成的文本
            references: 参考文本
            
        Returns:
            EvaluationResult
        """
        model = self._get_model()
        
        # 编码
        pred_emb = model.encode([predictions])
        ref_emb = model.encode([references])
        
        # 计算余弦相似度
        from sklearn.metrics.pairwise import cosine_similarity
        similarity = cosine_similarity(pred_emb, ref_emb)[0][0]
        
        return EvaluationResult(
            metric_name=self.name,
            value=float(similarity),
            metadata={"model": self.model_name}
        )


class LLMJudgeMetric(BaseMetric):
    """使用 LLM 作为评判器（评估答案质量）"""
    
    def __init__(
        self, 
        llm_func,
        aspect: str = "overall",
        prompt_template: Optional[str] = None
    ):
        """
        初始化 LLM 评判器
        
        Args:
            llm_func: LLM 函数
            aspect: 评价维度（faithfulness, relevance, coherence, completeness, overall）
            prompt_template: 自定义评价提示模板
        """
        super().__init__(name=f"LLM-Judge-{aspect}", aspect=aspect)
        self.llm_func = llm_func
        self.aspect = aspect
        self.prompt_template = prompt_template or self._get_default_prompt()
    
    def _get_default_prompt(self) -> str:
        """获取默认评价提示"""
        prompts = {
            "faithfulness": """
You are an expert evaluator. Given a QUESTION, CONTEXT, and ANSWER, rate how faithful the answer is to the context.
Faithfulness means the answer only contains information from the context and doesn't add unsupported claims.

QUESTION: {question}

CONTEXT: {context}

ANSWER: {answer}

Rate the faithfulness on a scale of 1-5:
1 = Completely unfaithful (contradicts or ignores context)
2 = Mostly unfaithful
3 = Partially faithful
4 = Mostly faithful
5 = Completely faithful (only uses information from context)

Output ONLY a single number (1-5) without explanation.""",

            "relevance": """
You are an expert evaluator. Given a QUESTION and ANSWER, rate how relevant the answer is to the question.

QUESTION: {question}

ANSWER: {answer}

Rate the relevance on a scale of 1-5:
1 = Completely irrelevant
2 = Mostly irrelevant
3 = Partially relevant
4 = Mostly relevant
5 = Completely relevant (directly answers the question)

Output ONLY a single number (1-5) without explanation.""",

            "coherence": """
You are an expert evaluator. Given an ANSWER, rate how coherent and well-structured it is.

ANSWER: {answer}

Rate the coherence on a scale of 1-5:
1 = Incoherent (confusing, contradictory)
2 = Mostly incoherent
3 = Partially coherent
4 = Mostly coherent
5 = Completely coherent (clear, logical flow)

Output ONLY a single number (1-5) without explanation.""",

            "completeness": """
You are an expert evaluator. Given a QUESTION and ANSWER, rate how complete the answer is.

QUESTION: {question}

ANSWER: {answer}

Rate the completeness on a scale of 1-5:
1 = Completely incomplete (missing key information)
2 = Mostly incomplete
3 = Partially complete
4 = Mostly complete
5 = Completely complete (comprehensive answer)

Output ONLY a single number (1-5) without explanation.""",

            "overall": """
You are an expert evaluator. Given a QUESTION, CONTEXT, and ANSWER, provide an overall quality rating.

QUESTION: {question}

CONTEXT: {context}

ANSWER: {answer}

Rate the overall quality on a scale of 1-5 considering:
- Faithfulness to context
- Relevance to question
- Coherence and clarity
- Completeness

Output ONLY a single number (1-5) without explanation."""
        }
        return prompts.get(self.aspect, prompts["overall"])
    
    async def compute(
        self, 
        predictions: str, 
        references: Dict[str, Any], 
        **kwargs
    ) -> EvaluationResult:
        """
        使用 LLM 评判答案质量
        
        Args:
            predictions: 生成的答案
            references: 包含 question, context, ground_truth 等的字典
            
        Returns:
            EvaluationResult
        """
        # 构造评价提示
        prompt = self.prompt_template.format(
            question=references.get("question", ""),
            context=references.get("context", ""),
            answer=predictions
        )
        
        # 调用 LLM
        try:
            response = await self.llm_func(prompt)
            
            # 提取分数（1-5）
            import re
            score_match = re.search(r'[1-5]', response)
            if score_match:
                score = int(score_match.group())
                normalized_score = score / 5.0  # 归一化到 0-1
            else:
                normalized_score = 0.0
            
            return EvaluationResult(
                metric_name=self.name,
                value=normalized_score,
                metadata={
                    "aspect": self.aspect,
                    "raw_score": score if score_match else None,
                    "llm_response": response
                }
            )
        except Exception as e:
            return EvaluationResult(
                metric_name=self.name,
                value=0.0,
                metadata={"error": str(e)}
            )


def create_generation_metrics(
    use_rouge: bool = True,
    use_bleu: bool = False,
    use_bertscore: bool = False,
    use_semantic_sim: bool = False
) -> List[BaseMetric]:
    """
    创建生成指标集合
    
    Args:
        use_rouge: 是否使用 ROUGE
        use_bleu: 是否使用 BLEU
        use_bertscore: 是否使用 BERTScore
        use_semantic_sim: 是否使用语义相似度
        
    Returns:
        List[BaseMetric]: 指标列表
    """
    metrics = []
    
    if use_rouge:
        metrics.extend([
            ROUGEMetric(rouge_type="rouge1"),
            ROUGEMetric(rouge_type="rouge2"),
            ROUGEMetric(rouge_type="rougeL"),
        ])
    
    if use_bleu:
        metrics.append(BLEUMetric())
    
    if use_bertscore:
        metrics.append(BERTScoreMetric())
    
    if use_semantic_sim:
        metrics.append(SemanticSimilarity())
    
    return metrics
