"""
IntelliDesk NLP Service
HuggingFace zero-shot classification for intent detection
+ Groq LLM for dynamic, intelligent responses
+ Unsplash image suggestions
"""

from typing import Tuple, Optional
from config import settings
from groq import Groq

# ── Groq client ────────────────────────────────────────────────────────────────
try:
    groq_client = Groq(api_key=settings.GROQ_API_KEY)
    GROQ_LOADED = True
    print("[NLP] Groq client initialised successfully.")
except Exception as e:
    GROQ_LOADED = False
    print(f"[NLP] Groq client failed to initialise: {e}")

# ── HuggingFace classifier ─────────────────────────────────────────────────────
try:
    from transformers import pipeline
    classifier = pipeline(
        "zero-shot-classification",
        model=settings.NLP_MODEL,
        device=-1
    )
    MODEL_LOADED = True
    print(f"[NLP] Model '{settings.NLP_MODEL}' loaded successfully.")
except Exception as e:
    MODEL_LOADED = False
    print(f"[NLP] Could not load HuggingFace model: {e}")
    print("[NLP] Falling back to keyword-based intent detection.")


# ── Intent labels & mappings ───────────────────────────────────────────────────
INTENT_LABELS = [
    "greeting or casual conversation",
    "password reset",
    "network connectivity issue",
    "software installation or update",
    "hardware malfunction",
    "access permission request",
    "general IT inquiry"
]

INTENT_TO_CATEGORY = {
    "greeting or casual conversation": "general",
    "password reset": "password",
    "network connectivity issue": "network",
    "software installation or update": "software",
    "hardware malfunction": "hardware",
    "access permission request": "access",
    "general IT inquiry": "general"
}

INTENT_PRIORITY = {
    "greeting or casual conversation": "low",
    "password reset": "low",
    "network connectivity issue": "medium",
    "software installation or update": "medium",
    "hardware malfunction": "high",
    "access permission request": "medium",
    "general IT inquiry": "low"
}

# ── Groq system prompt ─────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are IntelliDesk, a highly intelligent AI assistant created for a company's IT helpdesk platform. You have broad general knowledge but specialise deeply in IT support, technology, and computing.

Your personality:
- Friendly, confident, and conversational — like talking to a smart colleague, not a scripted bot
- You give direct, useful answers without unnecessary preamble
- You naturally use your IT expertise but can also answer general questions helpfully

How you respond:
- For IT problems: give clear, specific, step-by-step troubleshooting tailored to exactly what the user described. Address their specific OS, app, device, or error if mentioned.
- For follow-up questions: remember the full conversation context and build on it naturally
- For general questions: answer briefly and helpfully, then offer to help with any IT needs
- For casual chat: be warm and personable, keep it short
- Always use markdown formatting: **bold** for key terms, numbered lists for steps, `code` for commands or paths
- Explain WHY each step helps, not just what to do
- End IT troubleshooting responses with an encouraging line like "Let me know if that helps or if you want me to dig deeper!"
- Never say "I'm just an IT bot" or refuse non-IT questions — just answer naturally and briefly

You have deep expertise in: Windows, macOS, Linux, networking, Active Directory, Microsoft 365, Google Workspace, common enterprise software, cybersecurity basics, and general hardware troubleshooting."""

# ── Knowledge base (fallback) ──────────────────────────────────────────────────
KNOWLEDGE_BASE = {
    "greeting or casual conversation": {
        "response": (
            "Hello! I'm your AI IT support assistant. How can I help you today?\n\n"
            "You can ask me about:\n"
            "- Password resets\n"
            "- Network / WiFi issues\n"
            "- Software installation\n"
            "- Hardware problems\n"
            "- Access permissions\n\n"
            "Just describe your issue and I'll do my best to help!"
        )
    },
    "password reset": {
        "response": (
            "I can help you with your password reset! Here are the steps:\n\n"
            "1. Go to the login page and click 'Forgot Password'\n"
            "2. Enter your registered email address\n"
            "3. Check your inbox for the password reset link (also check spam folder)\n"
            "4. Click the link and set a new password\n"
            "5. Your new password must be at least 8 characters with one uppercase letter and one number\n\n"
            "If you still can't reset your password, the issue may need to be escalated to our IT team."
        )
    },
    "network connectivity issue": {
        "response": (
            "Let me help you troubleshoot your network connectivity issue:\n\n"
            "1. Check if your Wi-Fi is turned on and connected to the correct network\n"
            "2. Try turning Wi-Fi off and back on\n"
            "3. Restart your device\n"
            "4. If using ethernet, check that the cable is securely plugged in\n"
            "5. Run the network troubleshooter: Settings > Network & Internet > Network Troubleshooter\n"
            "6. If on VPN, try disconnecting and reconnecting\n\n"
            "If the issue persists, it may be a server-side problem that needs IT attention."
        )
    },
    "software installation or update": {
        "response": (
            "Here's how to handle software installation and update issues:\n\n"
            "1. Make sure you have admin rights on your machine\n"
            "2. Check if there's enough disk space (at least 2GB free recommended)\n"
            "3. Close all other applications before installing\n"
            "4. If updating, try uninstalling the current version first, then reinstalling\n"
            "5. Temporarily disable antivirus if it's blocking the installation\n"
            "6. Download the installer again in case the file was corrupted\n\n"
            "If you need a licensed software approval, a ticket will be created."
        )
    },
    "hardware malfunction": {
        "response": (
            "I'm sorry to hear about the hardware issue. Here are some initial checks:\n\n"
            "1. Restart your device completely (full shutdown, wait 30 seconds, power on)\n"
            "2. Check all cable connections are secure\n"
            "3. If it's a peripheral, try a different USB port\n"
            "4. Check Device Manager (Windows) or System Information (Mac)\n\n"
            "Hardware issues typically require hands-on support. A ticket will be created for our IT team."
        )
    },
    "access permission request": {
        "response": (
            "For access permission requests:\n\n"
            "1. Access requests must be approved by your manager or department head\n"
            "2. Please specify: the system you need access to, your role, and business justification\n"
            "3. Standard requests are processed within 24-48 business hours\n"
            "4. Elevated/admin access requires additional security clearance\n\n"
            "A ticket will be created so our IT administrators can process this."
        )
    },
    "general IT inquiry": {
        "response": (
            "Thank you for reaching out to IT support!\n\n"
            "For general inquiries:\n"
            "- Check our IT Knowledge Base for common solutions\n"
            "- Visit the self-service portal for account management\n"
            "- Review the IT policies page for guidelines\n\n"
            "If your question is specific, please provide more details and I'll try to assist. "
            "Otherwise, a support ticket will be created for you."
        )
    }
}

# ── Keyword fallback ───────────────────────────────────────────────────────────
KEYWORD_MAP = {
    "password reset": ["password", "reset", "forgot", "login fail", "can't log in", "locked out", "sign in"],
    "network connectivity issue": ["wifi", "wi-fi", "internet", "network", "connection", "disconnect", "vpn", "ethernet", "slow connection"],
    "software installation or update": ["install", "software", "update", "download", "application", "app", "program", "upgrade", "license", "slow", "sluggish", "performance", "lagging", "freezing", "hanging", "not responding", "windows", "crash", "loading"],
    "hardware malfunction": ["hardware", "screen", "monitor", "keyboard", "mouse", "printer", "laptop", "broken", "not working", "crash", "blue screen", "usb", "drive", "not showing", "device", "port", "headphones", "speaker", "webcam", "charger"],
    "access permission request": ["access", "permission", "role", "admin", "authorize", "restricted", "denied", "can't access"],
    "general IT inquiry": ["help", "question", "how to", "what is", "support", "issue", "problem"]
}


# ── Intent detection ───────────────────────────────────────────────────────────
def detect_intent_huggingface(text: str) -> Tuple[str, float]:
    result = classifier(text, INTENT_LABELS, multi_label=False)
    return result["labels"][0], round(result["scores"][0], 4)


def detect_intent_keyword(text: str) -> Tuple[str, float]:
    text_lower = text.lower()
    best_intent, best_score = "general IT inquiry", 0.0
    for intent, keywords in KEYWORD_MAP.items():
        matches = sum(1 for kw in keywords if kw in text_lower)
        if matches > 0:
            score = min(0.5 + (matches * 0.15), 0.95)
            if score > best_score:
                best_score, best_intent = score, intent
    return best_intent, round(best_score or 0.3, 4)


def detect_intent(text: str) -> Tuple[str, float]:
    if MODEL_LOADED:
        return detect_intent_huggingface(text)
    return detect_intent_keyword(text)


# ── Groq: IT response ──────────────────────────────────────────────────────────
def generate_groq_response(user_message: str, intent: str, history: list = None) -> Optional[str]:
    """Call Groq API to generate a dynamic IT support response."""
    user_prompt = (
        f"IT issue category: {intent}\n\n"
        f"User's exact message: \"{user_message}\"\n\n"
        f"Give a specific, tailored troubleshooting response that directly addresses "
        f"the details in their message. Do not give generic advice."
    )
    print(f"[GROQ] Sending to Groq — intent: {intent} | message: {user_message}")
    try:
        # Build messages array with history
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if history:
            for h in history[-6:]:  # last 6 exchanges to stay within token limits
                messages.append({"role": "user", "content": h["user"]})
                messages.append({"role": "assistant", "content": h["assistant"]})
        messages.append({"role": "user", "content": user_prompt})
        response = chat.choices[0].message.content.strip()
        print(f"[GROQ] Response received ({len(response)} chars)")
        return response
    except Exception as e:
        print(f"[GROQ] API error: {e}")
        return None


# ── Groq: off-topic response ───────────────────────────────────────────────────
def generate_off_topic_response(user_message: str) -> str:
    """Give a short friendly answer to non-IT questions, then redirect to IT."""
    try:
        chat = groq_client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=[
                {"role": "system", "content": (
                    "You are IntelliDesk, an IT support assistant with a friendly personality. "
                    "The user has asked something outside of IT support. "
                    "Give a very brief, helpful, natural answer to their question in 1-2 sentences max. "
                    "Then on a new line, gently remind them you're mainly here for IT support. "
                    "Keep the whole response under 4 sentences. Be warm and conversational, not robotic."
                )},
                {"role": "user", "content": user_message}
            ],
            temperature=0.8,
            max_tokens=150
        )
        return chat.choices[0].message.content.strip()
    except Exception as e:
        print(f"[GROQ] Off-topic response error: {e}")
        return "That's a bit outside my expertise! I'm mainly here for IT support — got any tech issues I can help with?"


# ── Groq: off-topic check ──────────────────────────────────────────────────────
def is_it_related(user_message: str) -> bool:
    """Ask Groq whether the message is IT-related. Returns True if it is."""
    if not GROQ_LOADED:
        return True
    try:
        result = groq_client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=[
                {"role": "system", "content": (
                    "You are a classifier. Respond with only 'yes' or 'no'.\n"
                    "Answer 'yes' if the message is related to IT support, technology, "
                    "computers, software, hardware, networks, passwords, or workplace tech.\n"
                    "Answer 'no' for everything else (e.g. social media, cooking, general chat, "
                    "homework, entertainment, random questions)."
                )},
                {"role": "user", "content": user_message}
            ],
            temperature=0,
            max_tokens=5
        )
        answer = result.choices[0].message.content.strip().lower()
        return answer.startswith("yes")
    except Exception as e:
        print(f"[GROQ] Off-topic check error: {e}")
        return True


# ── Groq: image query ──────────────────────────────────────────────────────────
def get_image_query(user_message: str, intent: str, is_it: bool) -> Optional[str]:
    """Ask Groq if an image would help. Returns a search query string or None."""
    if not GROQ_LOADED:
        return None
    try:
        result = groq_client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=[
                {"role": "system", "content": (
                    "You decide if showing an image would help answer the user's message.\n"
                    "If yes, reply with ONLY a short image search query (3-5 words, no quotes).\n"
                    "If no image is needed, reply with only the word: none\n"
                    "Show images for: hardware devices, network equipment, error screens, "
                    "step diagrams, or when user explicitly asks for a photo/image.\n"
                    "Do NOT show images for: password issues, permission requests, general text questions."
                )},
                {"role": "user", "content": f"User message: {user_message}\nIntent: {intent}"}
            ],
            temperature=0,
            max_tokens=20
        )
        query = result.choices[0].message.content.strip().lower()
        return None if query == "none" or not query else query
    except Exception as e:
        print(f"[GROQ] Image query error: {e}")
        return None


# ── Main response function ─────────────────────────────────────────────────────
def get_ai_response(user_message: str, intent: str, confidence: float, history: list = None) -> dict:
    threshold = settings.CONFIDENCE_THRESHOLD
    auto_ticket_intents = ["access permission request"]
    no_ticket_intents   = ["greeting or casual conversation"]

    # ── Follow-up detection ──
    followup_signals = ["how to", "how do", "what is", "what about", "narrow", "explain",
                        "in windows", "on mac", "more detail", "step by step", "tell me more",
                        "what if", "why", "which", "can you", "show me", "what does"]
    is_followup = bool(history) and any(s in user_message.lower() for s in followup_signals)

    # ── Greeting ──
    if intent in no_ticket_intents:
        return {
            "response": KNOWLEDGE_BASE["greeting or casual conversation"]["response"],
            "should_create_ticket": False,
            "category": "general",
            "priority": "low",
            "confident": True,
            "image_query": None
        }

    # ── Always answer with Groq if loaded (skip low-confidence ticket for follow-ups) ──
    # Only create ticket if: low confidence AND not a follow-up AND Groq not available
    if not GROQ_LOADED and confidence < threshold and not is_followup:
        return {
            "response": (
                "I'm not fully sure I understood your issue correctly. "
                "To make sure you get the right help, I've created a support ticket for you. "
                "Our IT team will review it and get back to you shortly.\n\n"
                f"What I understood: This might be related to '{intent}' "
                f"(confidence: {int(confidence * 100)}%)."
            ),
            "should_create_ticket": True,
            "category": INTENT_TO_CATEGORY.get(intent, "general"),
            "priority": "medium",
            "confident": False,
            "image_query": None
        }

    # ── Groq answers everything else ──
    groq_response = generate_groq_response(user_message, intent, history) if GROQ_LOADED else None
    img_query = get_image_query(user_message, intent, is_it=True) if GROQ_LOADED else None

    response_text = groq_response or KNOWLEDGE_BASE.get(
        intent, KNOWLEDGE_BASE["general IT inquiry"]
    )["response"]

    return {
        "response": response_text,
        "should_create_ticket": intent in auto_ticket_intents,
        "category": INTENT_TO_CATEGORY.get(intent, "general"),
        "priority": INTENT_PRIORITY.get(intent, "low"),
        "confident": True,
        "image_query": img_query
    }