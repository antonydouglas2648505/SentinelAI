import re

# Suspicious instruction patterns
THREAT_PATTERNS = {
    "instruction_override": [
        r"ignore (all )?(previous|prior|above) instructions",
        r"disregard (all )?(previous|prior|above) instructions",
        r"override (the )?(system|developer) instructions",
        r"forget (all )?(previous|prior) instructions"
    ],

    "system_prompt_extraction": [
        r"reveal (your )?(system prompt|hidden instructions)",
        r"show (me )?(your )?(system prompt|hidden prompt)",
        r"print (your )?(system prompt|hidden instructions)",
        r"what are your system instructions"
    ],

    "data_exfiltration": [
        r"send (all )?(the )?(data|information|secrets)",
        r"exfiltrate",
        r"upload .*?(passwords|credentials|secrets)",
        r"reveal .*?(passwords|credentials|api keys)"
    ],

    "unauthorized_actions": [
        r"execute .*?(command|script)",
        r"run .*?(shell|terminal) command",
        r"delete .*?(files|database|records)",
        r"access .*?(private|confidential) data"
    ]
}


UNTRUSTED_SOURCES = {
    "retrieved_document",
    "webpage",
    "email",
    "tool_output"
}


def analyze_security(text: str, source: str = "user_input"):

    detected_threats = []

    normalized_text = text.lower()

    for threat_type, patterns in THREAT_PATTERNS.items():

        for pattern in patterns:

            if re.search(pattern, normalized_text):

                detected_threats.append(threat_type)
                break

    is_untrusted = source in UNTRUSTED_SOURCES

    # Calculate heuristic risk score
    risk_score = 0

    if detected_threats:
        risk_score += min(len(detected_threats) * 30, 80)

    if is_untrusted and detected_threats:
        risk_score += 20

    risk_score = min(risk_score, 100)

    if risk_score >= 70:
        risk_level = "HIGH"
        recommended_action = "BLOCK"

    elif risk_score >= 30:
        risk_level = "MEDIUM"
        recommended_action = "FLAG_FOR_REVIEW"

    else:
        risk_level = "LOW"
        recommended_action = "ALLOW"

    return {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "detected_threats": detected_threats,
        "is_untrusted_source": is_untrusted,
        "recommended_action": recommended_action
    }