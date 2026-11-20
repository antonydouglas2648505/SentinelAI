from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):

    text: str = Field(
        ...,
        min_length=1,
        description="Text to analyze for security threats"
    )

    source: str = Field(
        default="user_input",
        description="Origin of the submitted text"
    )


class AnalyzeResponse(BaseModel):

    prediction: str
    benign_probability: float
    malicious_probability: float

    risk_score: int
    risk_level: str

    detected_threats: list[str]
    is_untrusted_source: bool

    recommended_action: str