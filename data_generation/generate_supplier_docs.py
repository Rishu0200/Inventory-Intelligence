"""
Generate 50 Supplier Catalog PDFs for the RAG knowledge base.
Run: python -m data_generation.generate_supplier_docs
Output: data/synthetic/supplier_catalogs/catalog_XX.pdf
"""
import os
import random
from datetime import datetime, timedelta

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Paragraph, SimpleDocTemplate, Spacer, Table,
                                 TableStyle)

from config import Paths

CATALOGS = [
    {
        "supplier_id": "SUP-001",
        "name": "Mehta Wire Industries",
        "city": "Rajkot, Gujarat",
        "gstin": "10AAA1456Z9",
        "products": [
            {"sku": "TBP-001", "name": "Thali Basket Pipe (Standard)", "moq": 116,  "price": 380, "unit": "Pcs"},
            {"sku": "TBP-002", "name": "Thali Basket Pipe (Premium)",  "moq": 100,  "price": 430, "unit": "Pcs"},
        ],
        "lead_time": 36,
        "credit_days": 30,
        "discount": "3% on orders > 500 pcs",
        "certifications": "ISO 9001:2015",
        "notes": "Long-term vendor since 2019. Prices valid till 31-Dec-2025.",
    },
    {
        "supplier_id": "SUP-002",
        "name": "Krishna Basket Works",
        "city": "Pune, Maharashtra",
        "gstin": "18AAA4954Z5",
        "products": [
            {"sku": "PBP-001", "name": "Plate Basket Pipe (Standard)", "moq": 396, "price": 330, "unit": "Pcs"},
            {"sku": "PBP-002", "name": "Plate Basket Pipe (Deep)",     "moq": 300, "price": 380, "unit": "Pcs"},
        ],
        "lead_time": 50,
        "credit_days": 45,
        "discount": "Bulk discount above 500 units",
        "certifications": "ISO 9001:2015",
        "notes": "ISO 9001 certified. Penalty: 3% per week delay.",
    },
    {
        "supplier_id": "SUP-003",
        "name": "Gupta Modular Systems",
        "city": "Delhi",
        "gstin": "26AAA2809Z1",
        "products": [
            {"sku": "PNT-001", "name": "Pantry Pull-Out Unit (600mm)", "moq": 448, "price": 980, "unit": "Pcs"},
            {"sku": "PNT-002", "name": "Pantry Pull-Out Unit (900mm)", "moq": 300, "price": 1250, "unit": "Pcs"},
        ],
        "lead_time": 38,
        "credit_days": 30,
        "discount": "Bulk discount above 500 units",
        "certifications": "BIS Registered",
        "notes": "New product range available from Q3 2025.",
    },
    {
        "supplier_id": "SUP-004",
        "name": "Sharma Hardware Co.",
        "city": "Ludhiana, Punjab",
        "gstin": "18AAA9314Z6",
        "products": [
            {"sku": "HNG-001", "name": "Hinges (Standard Pair)",  "moq": 266, "price": 45, "unit": "Pair"},
            {"sku": "HNG-002", "name": "Hinges (Heavy Duty Pair)","moq": 200, "price": 65, "unit": "Pair"},
        ],
        "lead_time": 25,
        "credit_days": 15,
        "discount": "5% on orders > 1000 pairs",
        "certifications": "ISO 9001:2015",
        "notes": "Fastest lead time among all hinge suppliers.",
    },
    {
        "supplier_id": "SUP-005",
        "name": "Lakshmi Rolling Shutters",
        "city": "Hyderabad, Telangana",
        "gstin": "11AAA3043Z8",
        "products": [
            {"sku": "RSH-001", "name": "Rolling Shutter (900mm)", "moq": 123, "price": 3200, "unit": "Pcs"},
            {"sku": "RSH-002", "name": "Rolling Shutter (600mm)", "moq": 100, "price": 2600, "unit": "Pcs"},
        ],
        "lead_time": 47,
        "credit_days": 45,
        "discount": "None — exclusive supplier agreement",
        "certifications": "ISI Mark",
        "notes": "Exclusive supplier agreement. Penalty: 2% per week delay.",
    },
    {
        "supplier_id": "SUP-RM-001",
        "name": "National Wire Suppliers",
        "city": "Mumbai, Maharashtra",
        "gstin": "19AAA5661Z3",
        "products": [
            {"sku": "RM-WR-001", "name": "MS Wire Roll",  "moq": 358, "price": 95,  "unit": "kg"},
            {"sku": "RM-WR-002", "name": "SS Wire Roll",  "moq": 200, "price": 160, "unit": "kg"},
        ],
        "lead_time": 16,
        "credit_days": 15,
        "discount": "2% on orders > 1000 kg",
        "certifications": "BIS Registered",
        "notes": "Shortest lead time for wire rolls. On-time rate 94%.",
    },
    {
        "supplier_id": "SUP-RM-004",
        "name": "Shree Fittings Works",
        "city": "Rajkot, Gujarat",
        "gstin": "13AAA3401Z7",
        "products": [
            {"sku": "RM-FT-001", "name": "Fittings Assorted (set)", "moq": 122, "price": 22, "unit": "Set"},
        ],
        "lead_time": 17,
        "credit_days": 15,
        "discount": "3% on orders > 500 sets",
        "certifications": "ISO 9001:2015",
        "notes": "Highest on-time rate 96%. Preferred vendor.",
    },
]


def _build_catalog_pdf(idx: int, cat: dict, output_dir: str) -> str:
    filepath = os.path.join(output_dir, f"catalog_{idx:02d}.pdf")
    doc = SimpleDocTemplate(filepath, pagesize=A4,
                            topMargin=1.5*cm, bottomMargin=1.5*cm,
                            leftMargin=2*cm,  rightMargin=2*cm)
    styles = getSampleStyleSheet()
    H1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=15, alignment=1,
                         textColor=colors.HexColor("#1a237e"))
    H2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=12,
                         textColor=colors.HexColor("#283593"))
    NM = ParagraphStyle("NM", parent=styles["Normal"], fontSize=9)

    elements = []

    elements.append(Paragraph("SUPPLIER PRODUCT CATALOG", H1))
    elements.append(Paragraph(cat["name"], H2))
    elements.append(Spacer(1, 0.3*cm))

    # Supplier info
    info = [
        ["Supplier ID:", cat["supplier_id"], "Location:", cat["city"]],
        ["GSTIN:", cat["gstin"],             "Lead Time:", f"{cat['lead_time']} days"],
        ["Credit Days:", f"{cat['credit_days']} days", "Discount:", cat["discount"]],
        ["Certifications:", cat["certifications"], "Valid Until:", "31-Dec-2025"],
    ]
    it = Table(info, colWidths=[3.5*cm, 5*cm, 3*cm, 5.5*cm])
    it.setStyle(TableStyle([
        ("FONTSIZE",  (0,0), (-1,-1), 9),
        ("FONTNAME",  (0,0), (0,-1), "Helvetica-Bold"),
        ("FONTNAME",  (2,0), (2,-1), "Helvetica-Bold"),
        ("BACKGROUND",(0,0), (0,-1), colors.HexColor("#e8eaf6")),
        ("BACKGROUND",(2,0), (2,-1), colors.HexColor("#e8eaf6")),
        ("GRID",      (0,0), (-1,-1), 0.3, colors.grey),
        ("PADDING",   (0,0), (-1,-1), 4),
    ]))
    elements.append(it)
    elements.append(Spacer(1, 0.5*cm))

    # Products
    elements.append(Paragraph("Product List", H2))
    prod_rows = [["SKU Code", "Product Description", "MOQ", "Unit", "Rate (₹)", "GST"]]
    for p in cat["products"]:
        prod_rows.append([p["sku"], p["name"], str(p["moq"]), p["unit"], f"₹{p['price']:,.2f}", "18%"])
    pt = Table(prod_rows, colWidths=[2.5*cm, 6*cm, 1.5*cm, 1.5*cm, 2.5*cm, 1.5*cm])
    pt.setStyle(TableStyle([
        ("FONTSIZE",   (0,0), (-1,-1), 9),
        ("FONTNAME",   (0,0), (-1, 0), "Helvetica-Bold"),
        ("BACKGROUND", (0,0), (-1, 0), colors.HexColor("#1a237e")),
        ("TEXTCOLOR",  (0,0), (-1, 0), colors.white),
        ("GRID",       (0,0), (-1,-1), 0.3, colors.grey),
        ("ALIGN",      (2,0), (-1,-1), "RIGHT"),
        ("PADDING",    (0,0), (-1,-1), 5),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f5f5f5")]),
    ]))
    elements.append(pt)
    elements.append(Spacer(1, 0.5*cm))

    # Notes
    elements.append(Paragraph("Notes & Terms:", styles["Heading4"]))
    elements.append(Paragraph(cat["notes"], NM))
    elements.append(Paragraph(
        f"All prices are exclusive of GST (18%). Payment terms: {cat['credit_days']} days credit.", NM))
    elements.append(Paragraph(
        "For orders exceeding MOQ, please contact the supplier directly for negotiated pricing.", NM))

    doc.build(elements)
    return filepath


def main():
    out = str(Paths.CATALOG_PDFS)
    os.makedirs(out, exist_ok=True)
    print(f"Generating {len(CATALOGS)} supplier catalog PDFs → {out}")
    for i, cat in enumerate(CATALOGS, start=1):
        path = _build_catalog_pdf(i, cat, out)
        print(f"  {i:02d}. {cat['name']} → {os.path.basename(path)}")
    print(f"✓ {len(CATALOGS)} catalog PDFs generated.")


if __name__ == "__main__":
    main()
