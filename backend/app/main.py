import csv
import io

from io import BytesIO
from xml.sax.saxutils import escape

from fastapi.responses import StreamingResponse
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    KeepTogether,
)

from fastapi.responses import StreamingResponse

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.detector import detector
from app.security import analyze_security
from app.schemas import AnalyzeRequest

from app.database import (
    initialize_database,
    save_analysis,
    get_analysis_logs,
    get_analysis_by_id,
    get_statistics,
    get_all_analysis_logs
)


app = FastAPI(
    title="SentinelAI API",
    description="AI-powered threat detection and prompt injection analysis",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Initialize database
initialize_database()


@app.get("/")
def home():

    return {
        "message": "Welcome to SentinelAI API",
        "status": "running"
    }


@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model": "RoBERTa",
        "security_engine": "active",
        "database": "connected"
    }


@app.post("/analyze")
def analyze(request: AnalyzeRequest):

    try:

        # ML classification
        ml_result = detector.predict(request.text)

        # Security analysis
        security_result = analyze_security(
            request.text,
            request.source
        )

        malicious_probability = ml_result[
            "malicious_probability"
        ]

        combined_score = max(
            security_result["risk_score"],
            int(malicious_probability)
        )

        if security_result["recommended_action"] == "BLOCK":
            final_action = "BLOCK"

        elif combined_score >= 50:
            final_action = "FLAG_FOR_REVIEW"

        else:
            final_action = "ALLOW"

        if combined_score >= 70:
            final_risk = "HIGH"

        elif combined_score >= 30:
            final_risk = "MEDIUM"

        else:
            final_risk = "LOW"

        result = {

            "prediction": ml_result["prediction"],

            "benign_probability": ml_result[
                "benign_probability"
            ],

            "malicious_probability": malicious_probability,

            "risk_score": combined_score,

            "risk_level": final_risk,

            "detected_threats": security_result[
                "detected_threats"
            ],

            "is_untrusted_source": security_result[
                "is_untrusted_source"
            ],

            "recommended_action": final_action
        }

        # Save analysis to SQLite
        save_analysis(
            request.text,
            request.source,
            result
        )

        return result

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


# Retrieve analysis history
@app.get("/logs")
def logs(limit: int = 50):

    if limit < 1 or limit > 500:
        raise HTTPException(
            status_code=400,
            detail="Limit must be between 1 and 500"
        )

    return {
        "logs": get_analysis_logs(limit)
    }


# Retrieve individual threat investigation
@app.get("/logs/{analysis_id}")
def get_analysis_details(analysis_id: int):

    if analysis_id < 1:
        raise HTTPException(
            status_code=400,
            detail="Analysis ID must be a positive integer"
        )

    analysis = get_analysis_by_id(analysis_id)

    if analysis is None:
        raise HTTPException(
            status_code=404,
            detail="Analysis record not found"
        )

    return {
        "message": "Analysis retrieved successfully",
        "analysis": analysis
    }


# Retrieve security statistics
@app.get("/stats")
def stats():

    return get_statistics()

# Export all security analysis records as CSV
@app.get("/export/csv")
def export_csv():

    try:
        logs = get_all_analysis_logs()

        output = io.StringIO()

        writer = csv.writer(output)

        # CSV column headers
        writer.writerow([
            "Analysis ID",
            "Timestamp",
            "Analyzed Text",
            "Source",
            "Prediction",
            "Benign Probability",
            "Malicious Probability",
            "Risk Score",
            "Risk Level",
            "Detected Threats",
            "Untrusted Source",
            "Recommended Action"
        ])

        # Prevent spreadsheet formula injection
        def safe_cell(value):
            value = str(value if value is not None else "")

            if value.lstrip().startswith(("=", "+", "-", "@", "\t", "\r")):
                return "'" + value

            return value

        # Add analysis records
        for log in logs:

            writer.writerow([
                log["id"],
                safe_cell(log["created_at"]),
                safe_cell(log["text"]),
                safe_cell(log["source"]),
                safe_cell(log["prediction"]),
                log["benign_probability"],
                log["malicious_probability"],
                log["risk_score"],
                safe_cell(log["risk_level"]),
                safe_cell(", ".join(log["detected_threats"])),
                log["is_untrusted_source"],
                safe_cell(log["recommended_action"])
            ])

        output.seek(0)

        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={
                "Content-Disposition":
                    'attachment; filename="sentinelai_security_report.csv"'
            }
        )

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Unable to generate security report."
        )

@app.get("/reports/{analysis_id}/pdf")
def generate_pdf_report(analysis_id: int):

    analysis = get_analysis_by_id(analysis_id)

    if not analysis:
        raise HTTPException(
            status_code=404,
            detail="Analysis record not found."
        )

    buffer = BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=20 * mm,
        leftMargin=20 * mm,
        topMargin=20 * mm,
        bottomMargin=20 * mm,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontSize=22,
        textColor=colors.HexColor("#172554"),
        alignment=TA_CENTER,
        spaceAfter=8,
    )

    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        fontSize=10,
        textColor=colors.HexColor("#64748b"),
        alignment=TA_CENTER,
        spaceAfter=20,
    )

    heading_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontSize=13,
        textColor=colors.HexColor("#1e40af"),
        spaceBefore=14,
        spaceAfter=8,
    )

    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["BodyText"],
        fontSize=9,
        leading=14,
        spaceAfter=6,
        wordWrap="CJK",
    )

    story = []

    def safe_paragraph(value):
        value = str(value if value is not None else "N/A")
        return Paragraph(
            escape(value).replace("\n", "<br/>"),
            body_style
        )

    # Report heading
    story.append(Paragraph("SENTINELAI", title_style))
    story.append(
        Paragraph(
            "SECURITY INVESTIGATION REPORT",
            subtitle_style
        )
    )

    story.append(
        Paragraph("1. Investigation Details", heading_style)
    )

    details = [
        ["Analysis ID", str(analysis.get("id", "N/A"))],
        ["Timestamp", str(analysis.get("created_at", "N/A"))],
        ["Source", str(analysis.get("source", "N/A"))],
        [
            "Untrusted Source",
            "Yes" if analysis.get("is_untrusted_source") else "No"
        ],
    ]

    details_table = Table(details, colWidths=[45 * mm, 115 * mm])

    details_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e2e8f0")),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#1e293b")),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("PADDING", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))

    story.append(details_table)

    # Analyzed content
    story.append(
        Paragraph("2. Analyzed Content", heading_style)
    )
    story.append(safe_paragraph(analysis.get("text", "N/A")))

    # ML classification
    story.append(
        Paragraph("3. Machine Learning Classification", heading_style)
    )

    classification = [
        ["Prediction", str(analysis.get("prediction", "N/A"))],
        [
            "Benign Probability",
            f"{analysis.get('benign_probability', 0):.2f}%"
        ],
        [
            "Malicious Probability",
            f"{analysis.get('malicious_probability', 0):.2f}%"
        ],
    ]

    classification_table = Table(
        classification,
        colWidths=[65 * mm, 95 * mm]
    )

    classification_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e2e8f0")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("PADDING", (0, 0), (-1, -1), 8),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
    ]))

    story.append(classification_table)

    # Security assessment
    story.append(
        Paragraph("4. Security Assessment", heading_style)
    )

    risk_details = [
        ["Risk Score", f"{analysis.get('risk_score', 0)}/100"],
        ["Risk Level", str(analysis.get("risk_level", "N/A"))],
        [
            "Recommended Action",
            str(analysis.get("recommended_action", "N/A"))
        ],
    ]

    risk_table = Table(risk_details, colWidths=[65 * mm, 95 * mm])

    risk_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e2e8f0")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("PADDING", (0, 0), (-1, -1), 8),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
    ]))

    story.append(risk_table)

    # Detected threats
    story.append(
        Paragraph("5. Detected Threats", heading_style)
    )

    threats = analysis.get("detected_threats", [])

    if threats:
        for threat in threats:
            story.append(
                Paragraph(
                    f"&#8226; {escape(str(threat))}",
                    body_style
                )
            )
    else:
        story.append(
            safe_paragraph("No rule-based threats detected.")
        )

    # Disclaimer
    story.append(Spacer(1, 15))

    story.append(
        Paragraph("Disclaimer", heading_style)
    )

    story.append(
        safe_paragraph(
            "This report is generated automatically by SentinelAI. "
            "Machine learning predictions and rule-based security "
            "checks are intended to assist security analysis and "
            "should not be treated as a guarantee that content is "
            "safe or malicious."
        )
    )

    # Generate PDF
    document.build(story)

    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="application/pdf",
        headers={
            "Content-Disposition": (
                f'attachment; filename="sentinelai_report_{analysis_id}.pdf"'
            )
        }
    )