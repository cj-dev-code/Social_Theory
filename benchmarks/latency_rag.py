# benchmarks/latency_rag.py
from __future__ import annotations
import time
from statistics import mean

from src.retrieval import get_retriever
from src.generate import generate, qdrant_client
from src import config


def retrieve_relevant_docs(question: str):
    retriever, _ = get_retriever(qdrant_client)
    return retriever.get_relevant_documents(question)


def generate_answer(question: str, docs):
    state = {"question": question, "context": docs}
    out = generate(state)
    return out["answer"]


def timed_answer(question: str):
    t0 = time.perf_counter()

    t_retr_start = time.perf_counter()
    docs = retrieve_relevant_docs(question)
    t_retr_end = time.perf_counter()

    t_llm_start = time.perf_counter()
    answer = generate_answer(question, docs)
    t_llm_end = time.perf_counter()

    t1 = time.perf_counter()

    metrics = {
        "total_ms": (t1 - t0) * 1000,
        "retrieval_ms": (t_retr_end - t_retr_start) * 1000,
        "llm_ms": (t_llm_end - t_llm_start) * 1000,
        "n_docs": len(docs),
    }
    return answer, metrics


def benchmark(question: str, runs: int = 5, warmup: int = 1):
    print(f"Benchmarking RAG latency for: {question!r}")
    print(f"Qdrant: {config.QDRANT_PATH} / {config.QDRANT_COLLECTION}")
    print(f"Ollama: {config.OLLAMA_MODEL_NAME}")

    for _ in range(warmup):
        timed_answer(question)

    totals, retrs, llms = [], [], []
    for i in range(runs):
        _, m = timed_answer(question)
        totals.append(m["total_ms"])
        retrs.append(m["retrieval_ms"])
        llms.append(m["llm_ms"])
        print(f"Run {i+1}: {m}")

    summary = {
        "runs": runs,
        "total_ms_avg": mean(totals),
        "retrieval_ms_avg": mean(retrs),
        "llm_ms_avg": mean(llms),
    }
    print("\n=== LATENCY SUMMARY ===")
    for k, v in summary.items():
        v_out = f"{v:.2f}" if isinstance(v, (float, int)) else v
        print(f"{k}: {v_out}")
    return summary


if __name__ == "__main__":
    queries = []

    queries.append(benchmark("Why do some authors call data feminism a form of justice work?"))
    queries.append(benchmark("How does data feminism redefine what counts as evidence?"))
    queries.append(benchmark("What role does intersectionality play in data science critiques?"))
    queries.append(benchmark("How do feminist scholars argue that power shows up in datasets?"))
    queries.append(benchmark("What is the relationship between care ethics and algorithm design?"))
    queries.append(benchmark("Why do feminist theorists emphasize context in data interpretation?"))
    queries.append(benchmark("How do feminist principles apply to machine learning workflows?"))
    queries.append(benchmark("What does design justice mean for AI systems?"))
    queries.append(benchmark("How can feminist thinking reduce model bias?"))
    queries.append(benchmark("Why is abstraction sometimes harmful according to data feminism?"))
    queries.append(benchmark("How does data feminism critique the idea of objectivity?"))
    queries.append(benchmark("What does standpoint theory contribute to algorithmic audits?"))
    queries.append(benchmark("How can we apply intersectional thinking to model evaluation?"))
    queries.append(benchmark("Why do feminist frameworks prioritize lived experience in analysis?"))
    queries.append(benchmark("How can care ethics improve responsible AI practices?"))
    queries.append(benchmark("What injustices arise when datasets erase marginalized groups?"))
    queries.append(benchmark("How does data feminism connect personal experience and computation?"))
    queries.append(benchmark("What does feminist epistemology say about who gets to produce knowledge?"))
    queries.append(benchmark("How can feminist critiques help redesign data collection practices?"))
    queries.append(benchmark("Why is transparency central to feminist approaches to AI?"))
    queries.append(benchmark("How does data feminism encourage rethinking default classifications?"))
    queries.append(benchmark("Why does data feminism critique scale as a value?"))
    queries.append(benchmark("How do feminist thinkers approach uncertainty in models?"))
    queries.append(benchmark("What strategies exist for resisting bias in algorithmic systems?"))
    queries.append(benchmark("How can feminist methods transform RAG pipelines ethically?"))
    queries.append(benchmark("Why do some feminist scholars reject 'neutral' model outputs?"))
    queries.append(benchmark("How is emotional labor relevant to data science teams?"))
    queries.append(benchmark("What does relationality mean in feminist data theory?"))
    queries.append(benchmark("How does data feminism intersect with abolitionist ethics?"))
    queries.append(benchmark("What feminist critiques exist of predictive policing algorithms?"))
    queries.append(benchmark("How can feminist theory illuminate hidden assumptions in NLP datasets?"))
    queries.append(benchmark("How do feminist thinkers critique quantification culture?"))
    queries.append(benchmark("What does 'data is never raw' mean in feminist STS?"))
    queries.append(benchmark("How does data feminism rethink what counts as 'the user'?"))
    queries.append(benchmark("Why do feminist scholars analyze absence as well as presence in datasets?"))
    queries.append(benchmark("How can design justice principles be applied to vector search systems?"))
    queries.append(benchmark("What does situated knowledge imply for model fine-tuning decisions?"))
    queries.append(benchmark("How does intersectionality apply to embedding bias?"))
    queries.append(benchmark("What feminist insights apply to explainability in AI?"))
    queries.append(benchmark("How can feminist critique help evaluate retrieval quality?"))
    queries.append(benchmark("How do feminist scholars argue against one-size-fits-all models?"))
    queries.append(benchmark("What does data feminism argue about power and default settings?"))
    queries.append(benchmark("How can feminist principles inform dataset documentation?"))
    queries.append(benchmark("Why do feminist thinkers analyze infrastructure in AI systems?"))
    queries.append(benchmark("How does data feminism challenge optimization-centric thinking?"))
    queries.append(benchmark("What are feminist critiques of benchmark-driven research cultures?"))
    queries.append(benchmark("How can feminist ethics guide the design of evaluation metrics?"))
    queries.append(benchmark("Why does data feminism argue for expanded notions of expertise?"))
    queries.append(benchmark("How can feminist theory inform responsible model deployment?"))

    total, retrieval, llm = 0,0,0
    for item in queries:
        _, t, r, l = item['runs'], item['total_ms_avg'], item['retrieval_ms_avg'], item['llm_ms_avg']
        total += t
        retrieval += r
        llm += l
    print('=== LATENCY STUDY ===')
    print("runs:", len(queries))
    print("total_ms_avg:", total/len(queries))
    print("retrieval_ms_avg:", retrieval/len(queries))
    print("llm_ms_avg", llm/len(queries))

