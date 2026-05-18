# swap_manager.py
# Manages dynamic model loading/unloading and context persistence.
# Core of the swap-moe pipeline.

import json
import os
import time
from pathlib import Path


CHECKPOINT_FILE = "checkpoint.json"
MODELS_DIR      = "./models"

MODEL_REGISTRY = {
    "router":     "Qwen/Qwen2.5-0.5B-Instruct",
    "medizin":    "mlx-community/SmolLM2-360M-Instruct-4bit",
    "technik":    "mlx-community/SmolLM2-360M-Instruct-4bit",
    "ernaehrung": "mlx-community/SmolLM2-360M-Instruct-4bit",
    "allgemein":  "mlx-community/SmolLM2-360M-Instruct-4bit",
}

ADAPTER_REGISTRY = {
    "router":     "./adapters/router",
    "medizin":    "./adapters/medizin",
    "technik":    "./adapters/technik",
    "ernaehrung": "./adapters/ernaehrung",
    "allgemein":  "./adapters/allgemein",
}


class SwapManager:
    """
    Handles model swapping, checkpoint saving and expert routing.
    Only one model is loaded in RAM at a time.
    """

    def __init__(self):
        self.active_model_name = None
        self.active_model      = None
        self.active_tokenizer  = None
        self.context           = {}
        self._load_checkpoint()
        print("[SwapManager] Ready.")
        print(f"[SwapManager] Context: {self.context}")

    # ── Private ───────────────────────────────────────────────

    def _load_checkpoint(self):
        """Load context from JSON file if it exists."""
        if os.path.exists(CHECKPOINT_FILE):
            with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
                self.context = json.load(f)
        else:
            self.context = {
                "domain":         None,
                "intent":         None,
                "last_model":     None,
                "history":        [],
                "session_active": False,
            }

    def _save_checkpoint(self):
        """Persist current context to JSON file."""
        with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
            json.dump(self.context, f, ensure_ascii=False, indent=2)

    def _unload_current(self):
        """Release active model from RAM."""
        if self.active_model is not None:
            print(f"[Swap] Unloading: {self.active_model_name}")
            self.active_model      = None
            self.active_tokenizer  = None
            self.active_model_name = None

    def _load_model(self, model_name: str):
        """Load a model and its adapter into RAM. Unloads current first."""
        if self.active_model_name == model_name:
            print(f"[Swap] {model_name} already loaded.")
            return

        self._unload_current()

        model_path   = MODEL_REGISTRY.get(model_name)
        adapter_path = ADAPTER_REGISTRY.get(model_name)

        if model_path is None:
            raise ValueError(f"Unknown model: '{model_name}'. "
                             f"Available: {list(MODEL_REGISTRY.keys())}")

        print(f"[Swap] Loading: {model_name}")
        start = time.time()

        try:
            from mlx_lm import load
            if adapter_path and os.path.exists(adapter_path):
                self.active_model, self.active_tokenizer = load(
                    model_path, adapter_path=adapter_path
                )
            else:
                self.active_model, self.active_tokenizer = load(model_path)
        except ImportError:
            # MLX not installed — use simulation mode
            print("[Swap] MLX not found — simulation mode active.")
            self.active_model     = f"SIMULATED:{model_name}"
            self.active_tokenizer = f"TOKENIZER:{model_name}"

        self.active_model_name = model_name
        print(f"[Swap] Loaded in {time.time() - start:.2f}s")

    # ── Public ────────────────────────────────────────────────

    def update_context(self, domain: str = None, intent: str = None,
                       user_input: str = None):
        """Update and save the current session context."""
        if domain:
            self.context["domain"] = domain
        if intent:
            self.context["intent"] = intent
        if user_input:
            self.context["history"].append(user_input)
            self.context["history"] = self.context["history"][-5:]
        self.context["session_active"] = True
        self._save_checkpoint()

    def route(self, user_input: str) -> dict:
        """Load router, classify input. Returns domain + intent."""
        print(f"\n[Router] Input: '{user_input}'")
        self._load_model("router")

        if not isinstance(self.active_model, str):
            from mlx_lm import generate
            system_prompt = (
                "You are a request classifier. Reply ONLY with JSON.\n"
                "Format: {\"domain\": \"...\", \"intent\": \"...\"}\n"
                "Domains: medizin, technik, ernaehrung, allgemein\n"
                "Intents: quick_info, research, personal_advice, how_to"
            )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_input},
            ]
            prompt = self.active_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            response = generate(
                self.active_model, self.active_tokenizer,
                prompt=prompt, max_tokens=50, verbose=False
            )
            try:
                result = json.loads(response.strip())
            except json.JSONDecodeError:
                result = {"domain": "allgemein", "intent": "quick_info"}
        else:
            result = self._simulate_routing(user_input)

        print(f"[Router] Result: {result}")
        return result

    def ask_expert(self, user_input: str, domain: str, intent: str) -> str:
        """Load expert model, generate answer with intent-aware system prompt."""
        print(f"\n[Expert] domain={domain} intent={intent}")
        self.update_context(domain=domain, intent=intent,
                            user_input=user_input)
        self._load_model(domain)

        intent_instructions = {
            "quick_info":      "Answer briefly in 1-2 sentences.",
            "research":        "Answer in detail with scientific background.",
            "personal_advice": "Answer empathetically. Recommend a doctor for medical issues.",
            "how_to":          "Explain step by step with examples.",
        }
        domain_context = {
            "medizin":    "You are a medical assistant.",
            "technik":    "You are an IT and tech expert.",
            "ernaehrung": "You are a nutrition advisor.",
            "allgemein":  "You are a helpful assistant.",
        }
        system_prompt = (
            f"{domain_context.get(domain, 'You are an assistant.')} "
            f"{intent_instructions.get(intent, 'Be helpful.')}"
        )

        messages = [{"role": "system", "content": system_prompt}]
        if self.context.get("history"):
            messages.append({
                "role": "system",
                "content": f"Context: {', '.join(self.context['history'][-2:])}"
            })
        messages.append({"role": "user", "content": user_input})

        if not isinstance(self.active_model, str):
            from mlx_lm import generate
            prompt = self.active_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            response = generate(
                self.active_model, self.active_tokenizer,
                prompt=prompt, max_tokens=300, verbose=False
            )
        else:
            response = self._simulate_expert(user_input, domain, intent)

        self.context["last_model"] = domain
        self._save_checkpoint()
        return response

    def process(self, user_input: str) -> str:
        """Full pipeline: route → load expert → answer → unload."""
        print(f"\n{'='*50}\nInput: {user_input}\n{'='*50}")
        routing = self.route(user_input)
        answer  = self.ask_expert(
            user_input,
            routing.get("domain", "allgemein"),
            routing.get("intent", "quick_info"),
        )
        self._unload_current()
        return answer

    def get_status(self) -> dict:
        """Return current manager state."""
        return {
            "active_model":      self.active_model_name,
            "context":           self.context,
            "models_available":  list(MODEL_REGISTRY.keys()),
        }

    # ── Simulation ────────────────────────────────────────────

    def _simulate_routing(self, user_input: str) -> dict:
        """Keyword-based fallback routing for testing without MLX."""
        text = user_input.lower()
        if any(w in text for w in ["vitamin", "krank", "mangel", "symptom"]):
            domain = "medizin"
        elif any(w in text for w in ["python", "code", "error", "bug"]):
            domain = "technik"
        elif any(w in text for w in ["kochen", "rezept", "kalorie", "essen"]):
            domain = "ernaehrung"
        else:
            domain = "allgemein"

        if any(w in text for w in ["forsche", "studie", "erkläre", "warum"]):
            intent = "research"
        elif any(w in text for w in ["ich", "mir", "mein", "habe"]):
            intent = "personal_advice"
        elif any(w in text for w in ["wie", "schritt", "anleitung"]):
            intent = "how_to"
        else:
            intent = "quick_info"

        return {"domain": domain, "intent": intent}

    def _simulate_expert(self, user_input: str, domain: str,
                          intent: str) -> str:
        """Demo answers for testing without MLX."""
        responses = {
            ("medizin",    "quick_info"):      "Adults need ~95mg Vitamin C per day (DGE).",
            ("medizin",    "research"):        "Vitamin C: 95mg/day. Scurvy below 10mg/day. Bioavailability drops above 200mg. Sources: DGE 2023, WHO.",
            ("medizin",    "personal_advice"): "This could indicate a deficiency. Please consult a doctor for a proper diagnosis.",
            ("technik",    "how_to"):          "1. Read the error. 2. Check variable types. 3. Use print() to debug.",
            ("technik",    "research"):        "TypeError occurs when an operator is applied to incompatible types. Example: '1' + 1.",
            ("ernaehrung", "quick_info"):      "You can make salad, stir-fry, soup or wraps with chicken breast.",
            ("allgemein",  "quick_info"):      "I am your local AI assistant. How can I help?",
        }
        return responses.get(
            (domain, intent),
            f"[Simulation] {domain}/{intent}: {user_input}"
        )
