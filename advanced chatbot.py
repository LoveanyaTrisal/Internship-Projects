#!/usr/bin/env python3
"""
ADVANCED Hybrid Q&A Chatbot: rules + ML + live knowledge. Pure Python, no installs.

Pipeline (first confident layer wins):
  1. RULES      math, unit conversion, time/date, name memory, teach command
  2. INTENT ML  Naive Bayes classifier -> small talk (greeting, thanks, bye...)
  3. LOCAL KB   TF-IDF + cosine similarity, typo-tolerant, learns via 'teach:'
  4. LLM        (optional) Claude API for open-ended questions, with chat memory.
                Enable by setting env var ANTHROPIC_API_KEY.
  5. WIKIPEDIA  free open-domain facts (needs internet, no key)
  6. FALLBACK   closest-question suggestions

Every reply is tagged with the layer that produced it, e.g. [wikipedia].
Commands: /sources on|off   /reset   /help

Run:  python hybrid_chatbot.py          (interactive)
      python hybrid_chatbot.py --demo   (scripted demo)
Teach it live:  teach: <question> => <answer>   (saved to kb_extra.json)
"""
import ast
import json
import math
import operator
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime
from difflib import get_close_matches

# --------------------------------------------------------------------------
# 1. NLP preprocessing
# --------------------------------------------------------------------------
STOPWORDS = set("""a an the is are was were be been am do does did of to in on at for with
and or but if so it its this that these those i you me my your we they he she can could
would should will shall please tell about what whats who how why when where which""".split())


def stem(w: str) -> str:
    """Tiny suffix stripper (Porter-lite)."""
    for suf in ("ingly", "edly", "ing", "ies", "ied", "ed", "es", "s", "ly"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: -len(suf)] + ("y" if suf in ("ies", "ied") else "")
    return w


def tokenize(text: str, keep_stop=False):
    toks = re.findall(r"[a-z0-9']+", text.lower().replace("'s", ""))
    return [stem(t) for t in toks if keep_stop or t not in STOPWORDS]


def ngrams(tokens):
    return tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]


# --------------------------------------------------------------------------
# 2. ML components (from scratch)
# --------------------------------------------------------------------------
class NaiveBayesIntent:
    """Multinomial Naive Bayes with additive (Lidstone) smoothing."""
    ALPHA = 0.1  # small alpha = sharper confidence on short utterances

    def fit(self, data):  # data: {intent: [examples]}
        self.priors, self.counts, self.totals = {}, {}, {}
        self.vocab = set()
        n = sum(len(v) for v in data.values())
        for intent, examples in data.items():
            c = Counter()
            for ex in examples:
                c.update(tokenize(ex, keep_stop=True))
            self.counts[intent] = c
            self.totals[intent] = sum(c.values())
            self.priors[intent] = math.log(len(examples) / n)
            self.vocab |= set(c)
        return self

    def predict(self, text):
        toks = [t for t in tokenize(text, keep_stop=True)]
        known = sum(t in self.vocab for t in toks)
        # reject open-domain questions: most words must be known small-talk words
        if not toks or known / len(toks) < 0.75:
            return None, 0.0
        V = len(self.vocab)
        scores = {}
        for intent in self.counts:
            s = self.priors[intent]
            for t in toks:
                if t in self.vocab:
                    s += math.log((self.counts[intent][t] + self.ALPHA) / (self.totals[intent] + self.ALPHA * V))
            scores[intent] = s
        m = max(scores.values())
        exp = {k: math.exp(v - m) for k, v in scores.items()}
        z = sum(exp.values())
        best = max(exp, key=exp.get)
        return best, exp[best] / z


class TfidfRetriever:
    def fit(self, docs):  # docs: list of (question, answer)
        self.docs = docs
        self.doc_tokens = [ngrams(tokenize(q)) for q, _ in docs]
        df = Counter()
        for toks in self.doc_tokens:
            df.update(set(toks))
        N = len(docs)
        self.idf = {t: math.log((1 + N) / (1 + d)) + 1 for t, d in df.items()}
        self.vecs = [self._vec(t) for t in self.doc_tokens]
        self.unigrams = [t for t in self.idf if "_" not in t]
        return self

    def _vec(self, toks):
        tf = Counter(toks)
        v = {t: c * self.idf[t] for t, c in tf.items() if t in self.idf}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {t: x / norm for t, x in v.items()}

    def _correct(self, toks):
        """Fuzzy-fix typos against known vocabulary ('pythn' -> 'python')."""
        out = []
        for t in toks:
            if t in self.idf or len(t) < 4:
                out.append(t)
            else:
                m = get_close_matches(t, self.unigrams, n=1, cutoff=0.78)
                out.append(m[0] if m else t)
        return out

    def query(self, text, k=3):
        toks = ngrams(self._correct(tokenize(text)))
        q = self._vec(toks)
        if not q:
            return []
        sims = []
        for i, v in enumerate(self.vecs):
            s = sum(w * v.get(t, 0.0) for t, w in q.items())
            sims.append((s, i))
        sims.sort(reverse=True)
        return [(s, self.docs[i][0], self.docs[i][1]) for s, i in sims[:k] if s > 0]


# --------------------------------------------------------------------------
# 3. Rule-based handlers
# --------------------------------------------------------------------------
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
        ast.USub: operator.neg, ast.FloorDiv: operator.floordiv}


def safe_eval(expr: str):
    """Evaluate arithmetic via AST (no eval())."""
    def ev(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            if isinstance(n.op, ast.Pow) and abs(ev(n.right)) > 100:
                raise ValueError("exponent too large")
            return _OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.operand))
        raise ValueError("unsupported")
    return ev(ast.parse(expr, mode="eval").body)


WORD_OPS = [(r"\bplus\b", "+"), (r"\bminus\b", "-"), (r"\b(times|multiplied by)\b", "*"),
            (r"\b(divided by|over)\b", "/"), (r"\bsquared\b", "**2"), (r"\bx\b", "*"), (r"\^", "**")]


# --------------------------------------------------------------------------
# 4. Data
# --------------------------------------------------------------------------
INTENTS = {
    "greeting": ["hello", "hi there", "hey", "good morning", "good evening", "howdy", "hi bot", "greetings"],
    "farewell": ["bye", "goodbye", "see you later", "talk to you soon", "i have to go", "quit", "exit", "cya"],
    "thanks": ["thanks", "thank you", "thanks a lot", "much appreciated", "cheers", "thx"],
    "how_are_you": ["how are you", "how are you doing", "how is it going", "whats up", "how do you feel"],
    "identity": ["who are you", "what are you", "what is your name", "are you a robot", "introduce yourself"],
    "capabilities": ["what can you do", "help", "how can you help me", "what do you know", "what are your features"],
}
RESPONSES = {
    "greeting": ["Hello! Ask me anything about AI, Python, or NLP."],
    "farewell": ["Goodbye! Come back anytime."],
    "thanks": ["You're welcome!"],
    "how_are_you": ["Running at full capacity, thanks! What would you like to know?"],
    "identity": ["I'm a hybrid chatbot: rules for precise tasks, machine learning for language understanding."],
    "capabilities": ["I can answer questions from my knowledge base, do math (try 'what is 12 * (3+4)'), "
                     "tell the time, remember your name, and learn new facts via "
                     "'teach: question => answer'."],
}

KNOWLEDGE = [
    ("What is machine learning?", "Machine learning is a field of AI where algorithms learn patterns from data instead of being explicitly programmed."),
    ("What is natural language processing?", "NLP is the area of AI that enables computers to understand, interpret and generate human language."),
    ("What is TF-IDF?", "TF-IDF weights a word by how often it appears in a document (TF) and how rare it is across all documents (IDF), highlighting distinctive terms."),
    ("What is tokenization?", "Tokenization splits text into smaller units such as words or subwords so models can process it."),
    ("What is stemming?", "Stemming chops word endings to reduce related words to a common root, e.g. 'running' -> 'run'."),
    ("What is lemmatization?", "Lemmatization maps words to their dictionary form using vocabulary and grammar, e.g. 'better' -> 'good'."),
    ("What is cosine similarity?", "Cosine similarity measures the angle between two vectors; values near 1 mean the texts point in the same direction, i.e. are similar."),
    ("What is Naive Bayes?", "Naive Bayes is a probabilistic classifier applying Bayes' theorem with the assumption that features are independent given the class."),
    ("What is a neural network?", "A neural network is a stack of layers of weighted connections that learns to map inputs to outputs via gradient descent."),
    ("What is deep learning?", "Deep learning uses neural networks with many layers to learn hierarchical representations from large datasets."),
    ("What is a transformer?", "A transformer is a neural architecture based on self-attention, the foundation of modern language models."),
    ("What is overfitting?", "Overfitting is when a model memorizes training data and performs poorly on new data. Regularization and more data help."),
    ("What is supervised learning?", "Supervised learning trains a model on labeled examples to predict labels for new inputs."),
    ("What is unsupervised learning?", "Unsupervised learning finds structure in unlabeled data, e.g. clustering or dimensionality reduction."),
    ("What is Python?", "Python is a high-level, readable programming language popular for data science, web development and automation."),
    ("How do I install a Python package?", "Use pip: run `pip install package_name` in your terminal."),
    ("What is a list comprehension in Python?", "A concise way to build lists, e.g. `[x*x for x in range(5)]`."),
    ("What is a chatbot?", "A chatbot is software that converses with people in natural language, using rules, machine learning, or both."),
    ("What is a hybrid chatbot?", "A hybrid chatbot combines deterministic rules for precise tasks with ML models for flexible language understanding."),
    ("What is named entity recognition?", "NER finds and classifies entities in text such as people, organizations, places and dates."),
    ("What is sentiment analysis?", "Sentiment analysis classifies text by emotional tone, such as positive, negative or neutral."),
]

EXTRA_FILE = "kb_extra.json"


# --------------------------------------------------------------------------
# 4b. Unit conversion rule
# --------------------------------------------------------------------------
UNITS = {  # unit -> (category, factor to base)
    "km": ("len", 1000), "kilometer": ("len", 1000), "kilometers": ("len", 1000),
    "m": ("len", 1), "meter": ("len", 1), "meters": ("len", 1),
    "cm": ("len", 0.01), "mm": ("len", 0.001),
    "mile": ("len", 1609.344), "miles": ("len", 1609.344),
    "ft": ("len", 0.3048), "feet": ("len", 0.3048), "foot": ("len", 0.3048),
    "inch": ("len", 0.0254), "inches": ("len", 0.0254),
    "kg": ("mass", 1000), "g": ("mass", 1), "gram": ("mass", 1), "grams": ("mass", 1),
    "lb": ("mass", 453.592), "lbs": ("mass", 453.592), "pound": ("mass", 453.592),
    "pounds": ("mass", 453.592), "oz": ("mass", 28.3495), "ounce": ("mass", 28.3495),
    "l": ("vol", 1), "liter": ("vol", 1), "liters": ("vol", 1), "ml": ("vol", 0.001),
    "gallon": ("vol", 3.78541), "gallons": ("vol", 3.78541),
}


def convert_units(text):
    m = re.search(r"([-\d.]+)\s*([a-z]+)\s+(?:to|in|into)\s+([a-z]+)", text.lower())
    if not m:
        return None
    val, a, b = m.groups()
    try:
        val = float(val)
    except ValueError:
        return None
    temps = {"c", "f", "k", "celsius", "fahrenheit", "kelvin"}
    if a in temps and b in temps:
        a, b = a[0], b[0]
        c = val if a == "c" else (val - 32) * 5 / 9 if a == "f" else val - 273.15
        r = c if b == "c" else c * 9 / 5 + 32 if b == "f" else c + 273.15
        return f"{val:g} deg {a.upper()} = {r:.2f} deg {b.upper()}"
    if a in UNITS and b in UNITS and UNITS[a][0] == UNITS[b][0]:
        r = val * UNITS[a][1] / UNITS[b][1]
        return f"{val:g} {a} = {r:.6g} {b}"
    return None


# --------------------------------------------------------------------------
# 4c. Open-domain knowledge: Wikipedia + optional LLM (stdlib only)
# --------------------------------------------------------------------------
UA = {"User-Agent": "HybridChatbot/2.0 (educational project)"}


def _http_json(url, data=None, headers=None, timeout=10):
    req = urllib.request.Request(url, data=data, headers={**UA, **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


class WikipediaSearch:
    """Looks up the best-matching article and returns its first sentences."""
    LEAD = re.compile(r"^(what|who|whom|where|when|which)\s+(is|are|was|were|does|do)\s+(a|an|the)?\s*|"
                      r"^(tell me about|explain|define|describe|give me info on|information on)\s+(a|an|the)?\s*", re.I)

    def __init__(self):
        self.cache = {}

    def topic(self, question):
        t = question.strip().rstrip("?.! ")
        t = self.LEAD.sub("", t)
        return t.strip() or question

    def lookup(self, question, sentences=3):
        topic = self.topic(question)
        if topic.lower() in self.cache:
            return self.cache[topic.lower()]
        result = None
        try:
            q = urllib.parse.urlencode({"action": "query", "list": "search", "srsearch": topic,
                                        "format": "json", "srlimit": 4})
            hits = _http_json("https://en.wikipedia.org/w/api.php?" + q)["query"]["search"]
            want = set(tokenize(topic))
            for h in hits:
                title = h["title"]
                overlap = want & set(tokenize(title))
                if want and not overlap:
                    continue
                s = _http_json("https://en.wikipedia.org/api/rest_v1/page/summary/"
                               + urllib.parse.quote(title.replace(" ", "_")))
                text = s.get("extract", "")
                if not text or "may refer to" in text or s.get("type") == "disambiguation":
                    continue
                parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
                result = (" ".join(parts[:sentences]), title)
                break
        except (urllib.error.URLError, KeyError, ValueError, TimeoutError, OSError):
            result = "OFFLINE"
        if result and result != "OFFLINE":
            self.cache[topic.lower()] = result
        return result


class LLMBackend:
    """Optional: Claude via the plain HTTPS API (no SDK needed)."""
    SYSTEM = ("You are a helpful, accurate assistant inside a hybrid chatbot. "
              "Answer concisely (max ~120 words). If unsure, say so.")

    def __init__(self):
        self.key = os.environ.get("ANTHROPIC_API_KEY")
        self.model = os.environ.get("CHATBOT_MODEL", "claude-sonnet-5-5")

    @property
    def enabled(self):
        return bool(self.key)

    def ask(self, history):
        body = json.dumps({"model": self.model, "max_tokens": 400, "system": self.SYSTEM,
                           "messages": history}).encode()
        try:
            r = _http_json("https://api.anthropic.com/v1/messages", data=body, timeout=30,
                           headers={"x-api-key": self.key, "anthropic-version": "2023-06-01",
                                    "content-type": "application/json"})
            return "".join(b.get("text", "") for b in r.get("content", [])).strip() or None
        except (urllib.error.URLError, ValueError, TimeoutError, OSError):
            return None


# --------------------------------------------------------------------------
# 5. The hybrid bot
# --------------------------------------------------------------------------
class HybridBot:
    INTENT_MIN = 0.60       # NB confidence needed to trust small-talk intent
    RETRIEVE_MIN = 0.35     # cosine score to answer directly
    RETRIEVE_SOFT = 0.15    # below this we give up; between = "did you mean"

    def __init__(self, extra_file=EXTRA_FILE):
        self.extra_file = extra_file
        self.kb = list(KNOWLEDGE) + self._load_extra()
        self.intent = NaiveBayesIntent().fit(INTENTS)
        self.retriever = TfidfRetriever().fit(self.kb)
        self.memory = {"name": None, "last_q": None}
        self.turn = 0
        self.wiki = WikipediaSearch()
        self.llm = LLMBackend()
        self.history = []          # chat memory for the LLM
        self.source = "none"
        self.show_sources = True

    # ---- persistence / learning
    def _load_extra(self):
        try:
            with open(self.extra_file) as f:
                return [tuple(x) for x in json.load(f)]
        except (OSError, ValueError):
            return []

    def teach(self, q, a):
        self.kb.append((q, a))
        self.retriever.fit(self.kb)
        try:
            with open(self.extra_file, "w") as f:
                json.dump(self.kb[len(KNOWLEDGE):], f, indent=2)
        except OSError:
            pass

    # ---- layer 1: rules
    def _rules(self, text):
        t = text.strip()
        low = t.lower()

        m = re.match(r"teach\s*:\s*(.+?)\s*=>\s*(.+)", t, re.I | re.S)
        if m:
            self.teach(m.group(1).strip(), m.group(2).strip())
            return "Got it, I've learned that."

        m = re.search(r"\bmy name is ([a-z][a-z'-]*)", low) or re.search(r"\bcall me ([a-z][a-z'-]*)", low)
        if m:
            self.memory["name"] = m.group(1).capitalize()
            return f"Nice to meet you, {self.memory['name']}!"
        if re.search(r"\b(what'?s|what is|do you know) my name\b", low):
            n = self.memory["name"]
            return f"Your name is {n}." if n else "I don't know yet. Tell me with 'my name is ...'."

        if re.search(r"\bwhat('?s| is) the time\b|\bcurrent time\b|\bwhat time is it\b", low):
            return "It's " + datetime.now().strftime("%H:%M:%S") + "."
        if re.search(r"\b(today'?s date|what('?s| is) the date|what day is (it|today))\b", low):
            return "Today is " + datetime.now().strftime("%A, %B %d, %Y") + "."

        # math: normalise words, then require digits and an operator
        expr = low
        for pat, rep in WORD_OPS:
            expr = re.sub(pat, rep, expr)
        expr = re.sub(r"\b(what is|whats|calculate|compute|solve|equals?)\b|[?=]", " ", expr)
        expr = re.sub(r"(?<=\d),(?=\d{3})", "", expr).strip()
        if re.fullmatch(r"[\d\s.+\-*/%()]+", expr) and re.search(r"\d", expr) and re.search(r"[+\-*/%]", expr):
            try:
                r = safe_eval(expr)
                return f"{round(r, 6):g}" if isinstance(r, float) else str(r)
            except ZeroDivisionError:
                return "Division by zero is undefined."
            except Exception:
                return "I couldn't evaluate that expression."
        conv = convert_units(low)
        if conv:
            return conv
        return None

    # ---- main entry
    def respond(self, text):
        reply = self._route(text)
        self.history += [{"role": "user", "content": text}, {"role": "assistant", "content": reply}]
        self.history = self.history[-12:]
        return reply

    def _command(self, text):
        t = text.strip().lower()
        if t == "/reset":
            self.history, self.memory["last_q"] = [], None
            return "Conversation memory cleared."
        if t.startswith("/sources"):
            self.show_sources = not t.endswith("off")
            return f"Source tags {'on' if self.show_sources else 'off'}."
        if t == "/help":
            return ("Ask anything. I do math, unit conversion, time/date, small talk, my built-in KB, "
                    "Wikipedia lookups" + (", and Claude for open-ended questions." if self.llm.enabled
                                           else ". Set ANTHROPIC_API_KEY for open-ended AI answers."))
        return None

    def _route(self, text):
        self.turn += 1
        self.source = "none"
        if not text.strip():
            return "Say something and I'll try to help!"
        cmd = self._command(text)
        if cmd:
            self.source = "command"
            return cmd

        ans = self._rules(text)                                   # 1 rules
        if ans:
            self.source = "rules"
            return ans

        hits = self.retriever.query(text)
        top = hits[0][0] if hits else 0.0
        intent, conf = self.intent.predict(text)                  # 2 intent

        if intent and conf >= self.INTENT_MIN and top < self.RETRIEVE_MIN:
            self.source = "ml-intent"
            reply = RESPONSES[intent][0]
            if intent == "greeting" and self.memory["name"]:
                reply = f"Hello again, {self.memory['name']}! " + reply.split("! ", 1)[-1]
            return reply

        if hits and top >= self.RETRIEVE_MIN:                     # 3 local KB
            self.source = "local-kb"
            self.memory["last_q"] = hits[0][1]
            return hits[0][2]

        if self.llm.enabled:                                      # 4 LLM
            out = self.llm.ask(self.history + [{"role": "user", "content": text}])
            if out:
                self.source = "llm"
                return out

        w = self.wiki.lookup(text)                                # 5 Wikipedia
        if w and w != "OFFLINE":
            self.source = "wikipedia"
            return f"{w[0]}\n(Source: Wikipedia - {w[1]})"

        if hits and top >= self.RETRIEVE_SOFT:                    # 6 fallback
            self.source = "suggest"
            sugg = "; ".join(f"'{q}'" for _, q, _ in hits[:3])
            return f"I'm not sure I understood. Did you mean: {sugg}?"
        self.source = "fallback"
        if w == "OFFLINE":
            return ("I couldn't reach the internet to look that up. Check your connection, "
                    "or teach me with 'teach: question => answer'.")
        return ("I couldn't find that. Try rephrasing, or teach me with "
                "'teach: question => answer'.")


# --------------------------------------------------------------------------
def demo():
    bot = HybridBot(extra_file=os.devnull)
    script = ["hi", "my name is Sam", "convert 10 km to miles", "100 f to c", "what is 12 * (3 + 4)",
              "What is TF idf?", "who was Marie Curie", "tell me about the Eiffel Tower",
              "what is quantum entanglement", "thanks"]
    for s in script:
        r = bot.respond(s)
        print(f"You: {s}\nBot: {r}  [{bot.source}]\n")


def main():
    if "--demo" in sys.argv:
        return demo()
    bot = HybridBot()
    mode = "Claude + Wikipedia" if bot.llm.enabled else "Wikipedia (set ANTHROPIC_API_KEY for AI answers)"
    print(f"Advanced Chatbot ready. Open-domain mode: {mode}. /help for commands.\n")
    while True:
        try:
            s = input("You: ")
        except (EOFError, KeyboardInterrupt):
            break
        reply = bot.respond(s)
        tag = f"  [{bot.source}]" if bot.show_sources else ""
        print(f"Bot: {reply}{tag}\n")
        if bot.source == "ml-intent" and bot.intent.predict(s)[0] == "farewell":
            break


if __name__ == "__main__":
    main()
