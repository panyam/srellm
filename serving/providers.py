import os, re, random
from typing import Dict

def _extract_user_issue(prompt: str) -> str:
    m = re.search(r"User issue:\s*(.+)$", prompt, flags=re.IGNORECASE | re.DOTALL)
    return m.group(1).strip() if m else prompt

class LocalMockProvider:
    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        text = _extract_user_issue(prompt).lower()

        candidates = []
        if "crashloop" in text or "pods restarting" in text:
            candidates = [
                "Run: kubectl describe; check probes; rollback image.",
                "Run: check probes; kubectl logs -p; rollback image.",
            ]
        elif "disk" in text or "no space left" in text or "95%" in text:
            candidates = [
                "Run: df -h; prune logs; enable logrotate.",
                "Run: df -h; clear /tmp and archived logs; set up logrotate.",
            ]
        elif "config" in text or "cache" in text or "flags not taking effect" in text:
            candidates = [
                "Run: check config version; purge cache; rolling restart stateless.",
                "Run: compare config store vs service; purge cache; restart stateless first.",
            ]
        else:
            candidates = ["Gather logs; check recent deploy; verify health checks."]

        if temperature and temperature > 0:
            return random.choice(candidates)
        return candidates[0]

class OpenRouterProvider:
    def __init__(self, model="meta-llama/llama-3.1-8b-instruct"):
        import httpx
        self.httpx = httpx
        self.model = model
        self.api_key = os.getenv("OPENROUTER_API_KEY")

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        if not self.api_key:
            raise RuntimeError("Set OPENROUTER_API_KEY or use LocalMockProvider.")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "http://localhost",
            "X-Title": "sre-llm-lab"
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature
        }
        resp = self.httpx.post("https://openrouter.ai/api/v1/chat/completions",
                               headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()

