# src/optimization/ga_reporter.py
"""
Generate PDF report for GA results.
"""

import os
from datetime import datetime
from typing import Any, Dict, List

try:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
    )
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False


class GAReporter:
    """Generate PDF report for GA results."""

    def __init__(self, output_dir: str = "ga_results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def generate_report(
        self,
        model_name: str,
        best_params: Dict[str, Any],
        best_fitness: float,
        history: List[Dict[str, Any]],
        hardware_profile: Dict[str, Any],
        output_path: str = None,
    ) -> str:
        """Generate PDF report."""
        if output_path is None:
            output_path = os.path.join(
                self.output_dir, f"ga_report_{model_name}.pdf"
            )

        if not HAS_REPORTLAB:
            return self._generate_json(
                model_name, best_params, best_fitness,
                history, hardware_profile, output_path
            )

        doc = SimpleDocTemplate(
            output_path, pagesize=A4,
            rightMargin=50, leftMargin=50, topMargin=50, bottomMargin=50,
        )

        styles = getSampleStyleSheet()
        title = ParagraphStyle(
            "T", parent=styles["Heading1"], fontSize=20,
            alignment=TA_CENTER, spaceAfter=20,
        )
        h1 = ParagraphStyle(
            "H1", parent=styles["Heading2"], fontSize=14, spaceAfter=10,
        )
        h2 = ParagraphStyle(
            "H2", parent=styles["Heading3"], fontSize=12, spaceAfter=8,
        )
        body = styles["Normal"]

        story = []

        # Title
        story.append(Paragraph(f"GA Report — {model_name.upper()}", title))
        story.append(Paragraph(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            body,
        ))
        story.append(Spacer(1, 20))

        # Summary
        story.append(Paragraph("1. Summary", h1))
        story.append(Paragraph(
            f"<b>Best Fitness (Linear Probe Accuracy):</b> {best_fitness:.4f}<br/>"
            f"<b>Total Generations:</b> {len(history)}<br/>"
            f"<b>Total Trials:</b> {len(history) * 12 if history else 0}<br/>",
            body,
        ))
        story.append(Spacer(1, 15))

        # Hardware
        story.append(Paragraph("2. Hardware Profile", h1))
        hw_data = [["Component", "Value"]]
        for k, v in hardware_profile.items():
            hw_data.append([str(k), str(v)])
        t = Table(hw_data, colWidths=[2.5 * inch, 4 * inch])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        story.append(t)
        story.append(Spacer(1, 15))

        # Best Params
        story.append(Paragraph("3. Best Hyperparameters", h1))
        bp_data = [["Parameter", "Value"]]
        for k, v in best_params.items():
            bp_data.append([str(k), str(v)])
        t = Table(bp_data, colWidths=[2.5 * inch, 4 * inch])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#16a085")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ]))
        story.append(t)
        story.append(Spacer(1, 15))

        # Evolution history
        story.append(PageBreak())
        story.append(Paragraph("4. Evolution History", h1))
        hist_data = [["Gen", "Best", "Mean", "Std", "Worst"]]
        for h in history:
            hist_data.append([
                str(h["generation"]),
                f"{h['best_fitness']:.4f}",
                f"{h['mean_fitness']:.4f}",
                f"{h['std_fitness']:.4f}",
                f"{h['worst_fitness']:.4f}",
            ])
        t = Table(hist_data, colWidths=[1 * inch, 1.5 * inch, 1.5 * inch, 1.5 * inch, 1.5 * inch])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ]))
        story.append(t)

        doc.build(story)
        print(f"✅ PDF saved: {output_path}")
        return output_path

    def _generate_json(self, *args, **kwargs):
        import json
        model_name = args[0]
        out = os.path.join(self.output_dir, f"ga_report_{model_name}.json")
        data = {
            "model_name": model_name,
            "best_params": args[1],
            "best_fitness": args[2],
            "history": args[3],
            "hardware": args[4],
            "timestamp": datetime.now().isoformat(),
        }
        with open(out, "w") as f:
            json.dump(data, f, indent=4, default=str)
        print(f"✅ JSON saved: {out}")
        return out