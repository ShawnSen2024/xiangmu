"""
从服务日志中提取 AI 验配 API 请求数据，用于批量验证
"""
import re
import ast
from pathlib import Path

def extract_requests(log_path):
    """
    从日志文件提取所有 API 请求数据
    """
    pattern = r"Received request data: ({.*?})"

    with open(log_path, "r", encoding="utf-8") as f:
        content = f.read()

    matches = re.findall(pattern, content, re.DOTALL)
    results = []

    for i, match in enumerate(matches):
        try:
            data = ast.literal_eval(match)
            chip = data.get("chip", "").lower()
            if "e7111v2" not in chip and "e7160sl" not in chip:
                continue
            results.append(data)
        except (ValueError, SyntaxError) as e:
            print(f"[line {i}] parse error: {e}")

    print(f"extracted {len(results)} requests from {len(matches)} matches")
    return results

if __name__ == "__main__":
    import json
    test = extract_requests("app.log.2026-05-22")
    if test:
        out = Path(__file__).resolve().parent / "outputs" / "test_cases.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(test, f, ensure_ascii=False, indent=2)
        print(f"saved {len(test)} cases to {out}")
