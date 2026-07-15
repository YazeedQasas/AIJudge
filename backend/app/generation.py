import re

CITATION_PATTERN = re.compile(r"\[(\d+)\]")


def build_prompt(question: str, chunks: list[dict]) -> str:
    """Build a grounded Arabic prompt: numbered sources, strict instructions, then the question.

    Instructions are written in Arabic (not English) so the model stays in Arabic and answers
    in Arabic. Source numbers stay as ASCII [1], [2], ... so citations match the numbered
    `sources` list returned to the frontend.
    """
    sources_block = "\n\n".join(
        f"[{i + 1}] (من {chunk['payload']['source']})\n{chunk['payload']['text']}"
        for i, chunk in enumerate(chunks)
    )
    return (
        "أنت قاضٍ آلي، مهمتك اقتراح الحكم الأنسب لقاضٍ حقيقي بناءً على القضية المطروحة "
        "والمصادر القانونية أدناه. أجب عن السؤال بالاعتماد على المصادر المرقّمة أدناه فقط، "
        "ولا تستعن بأي معرفة خارجة عمّا ورد فيها.\n\n"
        "لكل معلومة تذكرها في إجابتك، أشر إلى رقم المصدر بين قوسين معقوفين تماماً كما هو "
        "موضّح، هكذا [1].\n\n"
        "إذا كانت المصادر لا تتضمّن ما يكفي للإجابة عن السؤال، فصرّح بذلك بوضوح بدلاً من "
        "التخمين.\n\n"
        "يجب أن تكون إجابتك كاملةً باللغة العربية.\n\n"
        f"المصادر:\n{sources_block}\n\n"
        f"السؤال: {question}\n\n"
        "الإجابة:"
    )


def find_invalid_citations(answer: str, num_sources: int) -> list[int]:
    """Return citation numbers the model used that don't correspond to a real numbered source."""
    cited = {int(n) for n in CITATION_PATTERN.findall(answer)}
    return sorted(n for n in cited if n < 1 or n > num_sources)
