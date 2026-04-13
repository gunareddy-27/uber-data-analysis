from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from datetime import datetime
import io

def generate_live_events_pdf(events):
    """
    Generates a professional PDF document for Live Ride Events using reportlab.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    elements = []
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'MainTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor("#636EFA"),
        spaceAfter=20,
        alignment=1 # Center
    )
    
    # 1. Header
    elements.append(Paragraph("Uber Intelligence: Live Event Report", title_style))
    elements.append(Paragraph(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    elements.append(Spacer(1, 20))
    
    # 2. Summary
    elements.append(Paragraph(f"Total Events Ingested: {len(events)}", styles['Heading3']))
    elements.append(Spacer(1, 10))
    
    # 3. Table Data
    if not events:
        elements.append(Paragraph("No events detected in the current buffer.", styles['Normal']))
    else:
        # Prepare data for Table
        data = [["ID", "Timestamp", "Location", "Miles", "Category", "Purpose", "Status"]]
        for ev in events:
            data.append([
                ev.get('id', 'N/A'),
                ev.get('timestamp', 'N/A'),
                ev.get('location', 'N/A'),
                f"{ev.get('miles', 0)} mi",
                ev.get('category', 'N/A'),
                ev.get('purpose', 'N/A'),
                ev.get('status', 'N/A')
            ])
            
        t = Table(data, colWidths=[60, 70, 100, 50, 80, 80, 80])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#636EFA")),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,0), 12),
            ('BOTTOMPADDING', (0,0), (-1,0), 12),
            ('BACKGROUND', (0,1), (-1,-1), colors.beige),
            ('GRID', (0,0), (-1,-1), 1, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.ivory])
        ]))
        elements.append(t)
        
    # 4. Footer
    elements.append(Spacer(1, 40))
    elements.append(Paragraph("--- Confidential System Intelligence Log ---", styles['Normal']))
    
    doc.build(elements)
    buffer.seek(0)
    return buffer
