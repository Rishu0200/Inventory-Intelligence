"""
Generate 200 Purchase Order PDFs using real Uninox Houseware supplier and SKU data.
Run: python -m data_generation.generate_pos
Output: data/synthetic/purchase_orders/PO_XXXX.pdf
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

# ── Real data from supplier_directory.csv & supplier_terms.csv ───────────────

SUPPLIERS = [
    {"id": "SUP-001", "name": "Mehta Wire Industries",    "city": "Rajkot, Gujarat",          "gstin": "10AAA1456Z9", "contact": "Ramesh Mehta",   "lead": 36, "moq": 116, "credit": 30},
    {"id": "SUP-002", "name": "Krishna Basket Works",     "city": "Pune, Maharashtra",         "gstin": "18AAA4954Z5", "contact": "Suresh Kale",    "lead": 50, "moq": 396, "credit": 45},
    {"id": "SUP-003", "name": "Gupta Modular Systems",    "city": "Delhi",                     "gstin": "26AAA2809Z1", "contact": "Anil Gupta",     "lead": 38, "moq": 448, "credit": 30},
    {"id": "SUP-004", "name": "Sharma Hardware Co.",      "city": "Ludhiana, Punjab",          "gstin": "18AAA9314Z6", "contact": "Raj Sharma",     "lead": 25, "moq": 266, "credit": 15},
    {"id": "SUP-005", "name": "Lakshmi Rolling Shutters", "city": "Hyderabad, Telangana",      "gstin": "11AAA3043Z8", "contact": "Venkat Reddy",   "lead": 47, "moq": 123, "credit": 45},
    {"id": "SUP-006", "name": "Patel Chimney Solutions",  "city": "Surat, Gujarat",            "gstin": "28AAA1085Z8", "contact": "Hitesh Patel",   "lead": 37, "moq": 116, "credit": 30},
    {"id": "SUP-007", "name": "ATC Magic Corners",        "city": "Chennai, Tamil Nadu",       "gstin": "17AAA9737Z4", "contact": "Arjun T C",      "lead": 67, "moq": 292, "credit": 60},
    {"id": "SUP-RM-001","name":"National Wire Suppliers", "city": "Mumbai, Maharashtra",       "gstin": "19AAA5661Z3", "contact": "Deepak Joshi",   "lead": 16, "moq": 358, "credit": 15},
    {"id": "SUP-RM-004","name":"Shree Fittings Works",    "city": "Rajkot, Gujarat",           "gstin": "13AAA3401Z7", "contact": "Jayesh Modi",    "lead": 17, "moq": 122, "credit": 15},
    {"id": "SUP-RM-005","name":"Papercraf Packaging",     "city": "Nashik, Maharashtra",       "gstin": "12AAA2511Z1", "contact": "Prakash Jadhav", "lead": 34, "moq": 160, "credit": 30},
]

SUPPLIER_SKU = {
    "SUP-001":    [("TBP-001", "Thali Basket Pipe",          320, 420)],
    "SUP-002":    [("PBP-001", "Plate Basket Pipe",          280, 380)],
    "SUP-003":    [("PNT-001", "Pantry Unit",                850, 1150)],
    "SUP-004":    [("HNG-001", "Hinges (Pair)",               30,   65)],
    "SUP-005":    [("RSH-001", "Rolling Shutter Assembly",  2800, 4200)],
    "SUP-006":    [("CHM-001", "Chimney (90cm)",            3800, 5600)],
    "SUP-007":    [("MGC-001", "Magic Corner Unit",         4200, 6000)],
    "SUP-RM-001": [("RM-WR-001","Wire Roll MS (kg)",          85,  110),
                   ("RM-WR-002","Wire Roll SS (kg)",          140,  180)],
    "SUP-RM-004": [("RM-FT-001","Fittings Assorted (set)",    18,   30)],
    "SUP-RM-005": [("RM-CB-001","Couger Box Small (pcs)",      4,    8),
                   ("RM-CB-002","Couger Box Large (pcs)",      6,   12)],
}

BUYER = {
    "name":    "Aashi Enterprises (Uninox Houseware)",
    "address": "Plot 12, Industrial Area Phase II, Rohini, Delhi - 110041",
    "gstin":   "07AAXFA1234B1ZX",
    "phone":   "011-45678901",
    "email":   "purchase@uninox.in",
}

PAYMENT_TERMS = [
    "30 days net", "45 days net", "15 days net",
    "50% advance, balance on delivery", "LC at sight",
]


def _rand_date(start_yr=2022, end_yr=2025) -> datetime:
    start = datetime(start_yr, 1, 1)
    end   = datetime(end_yr, 5, 18)
    return start + timedelta(days=random.randint(0, (end - start).days))


def _build_po_pdf(po_num: int, output_dir: str) -> str:
    sup = random.choice(SUPPLIERS)
    skus = SUPPLIER_SKU.get(sup["id"], SUPPLIER_SKU["SUP-001"])
    sku_code, sku_name, p_lo, p_hi = random.choice(skus)

    po_date      = _rand_date()
    delivery_dt  = po_date + timedelta(days=sup["lead"] + random.randint(-5, 10))
    qty          = random.randint(sup["moq"], sup["moq"] * 4)
    unit_price   = round(random.uniform(p_lo, p_hi), 2)
    subtotal     = round(qty * unit_price, 2)
    gst          = round(subtotal * 0.18, 2)
    total        = round(subtotal + gst, 2)
    terms        = random.choice(PAYMENT_TERMS)
    po_number    = f"PO-{po_date.year}-{po_num:04d}"

    filepath = os.path.join(output_dir, f"PO_{po_num:04d}.pdf")
    doc = SimpleDocTemplate(filepath, pagesize=A4,
                            topMargin=1.5*cm, bottomMargin=1.5*cm,
                            leftMargin=2*cm,  rightMargin=2*cm)
    styles = getSampleStyleSheet()
    H1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=16, alignment=1,
                         spaceAfter=6, textColor=colors.HexColor("#1a237e"))
    NM = ParagraphStyle("NM", parent=styles["Normal"], fontSize=9)

    elements = []
    elements.append(Paragraph("PURCHASE ORDER", H1))
    elements.append(Spacer(1, 0.4*cm))

    # ── Meta row ──
    meta = [["PO Number:", po_number,       "PO Date:",       po_date.strftime("%d-%m-%Y")],
            ["Delivery By:", delivery_dt.strftime("%d-%m-%Y"), "Payment Terms:", terms]]
    mt = Table(meta, colWidths=[3*cm, 5.5*cm, 3*cm, 5.5*cm])
    mt.setStyle(TableStyle([
        ("FONTSIZE", (0,0), (-1,-1), 9),
        ("FONTNAME", (0,0), (0,-1), "Helvetica-Bold"),
        ("FONTNAME", (2,0), (2,-1), "Helvetica-Bold"),
        ("BACKGROUND", (0,0), (0,-1), colors.HexColor("#e8eaf6")),
        ("BACKGROUND", (2,0), (2,-1), colors.HexColor("#e8eaf6")),
        ("GRID", (0,0), (-1,-1), 0.3, colors.grey),
        ("PADDING", (0,0), (-1,-1), 4),
    ]))
    elements.append(mt)
    elements.append(Spacer(1, 0.4*cm))

    # ── Buyer / Supplier ──
    parties = [
        ["BUYER",                                  "SUPPLIER"],
        [BUYER["name"],                            sup["name"]],
        [BUYER["address"],                         sup["city"]],
        [f"GSTIN: {BUYER['gstin']}",               f"GSTIN: {sup['gstin']}"],
        [f"Contact: {BUYER['email']}",             f"Contact: {sup['contact']}"],
    ]
    pt = Table(parties, colWidths=[9*cm, 9*cm])
    pt.setStyle(TableStyle([
        ("FONTSIZE",   (0,0), (-1,-1), 9),
        ("FONTNAME",   (0,0), (-1, 0), "Helvetica-Bold"),
        ("BACKGROUND", (0,0), (-1, 0), colors.HexColor("#1a237e")),
        ("TEXTCOLOR",  (0,0), (-1, 0), colors.white),
        ("GRID",       (0,0), (-1,-1), 0.3, colors.grey),
        ("VALIGN",     (0,0), (-1,-1), "TOP"),
        ("PADDING",    (0,0), (-1,-1), 5),
    ]))
    elements.append(pt)
    elements.append(Spacer(1, 0.4*cm))

    # ── Items ──
    items = [["S.No", "SKU Code", "Description", "Qty", "Unit", "Rate (₹)", "Amount (₹)"],
             ["1",   sku_code,   sku_name,       str(qty), "Pcs",
              f"{unit_price:,.2f}", f"{subtotal:,.2f}"]]
    it = Table(items, colWidths=[1*cm, 2.5*cm, 5.5*cm, 1.5*cm, 1.5*cm, 2.5*cm, 3.5*cm])
    it.setStyle(TableStyle([
        ("FONTSIZE",   (0,0), (-1,-1), 9),
        ("FONTNAME",   (0,0), (-1, 0), "Helvetica-Bold"),
        ("BACKGROUND", (0,0), (-1, 0), colors.HexColor("#e8eaf6")),
        ("GRID",       (0,0), (-1,-1), 0.3, colors.grey),
        ("ALIGN",      (3,0), (-1,-1), "RIGHT"),
        ("PADDING",    (0,0), (-1,-1), 5),
    ]))
    elements.append(it)
    elements.append(Spacer(1, 0.2*cm))

    # ── Totals ──
    totals = [["", "", "", "", "", "Subtotal:",    f"₹{subtotal:,.2f}"],
              ["", "", "", "", "", "GST @ 18%:",   f"₹{gst:,.2f}"],
              ["", "", "", "", "", "TOTAL:",        f"₹{total:,.2f}"]]
    tt = Table(totals, colWidths=[1*cm, 2.5*cm, 5.5*cm, 1.5*cm, 1.5*cm, 2.5*cm, 3.5*cm])
    tt.setStyle(TableStyle([
        ("FONTSIZE",   (0,0), (-1,-1), 9),
        ("FONTNAME",   (-2,-1), (-1,-1), "Helvetica-Bold"),
        ("ALIGN",      (-2,0), (-1,-1), "RIGHT"),
        ("LINEABOVE",  (-2,-1), (-1,-1), 1, colors.black),
        ("PADDING",    (0,0), (-1,-1), 4),
    ]))
    elements.append(tt)
    elements.append(Spacer(1, 0.7*cm))

    # ── Terms ──
    elements.append(Paragraph("Terms & Conditions:", styles["Heading4"]))
    elements.append(Paragraph(f"1. Payment: {terms}.", NM))
    elements.append(Paragraph(
        f"2. Goods must be delivered by {delivery_dt.strftime('%d-%m-%Y')}. "
        f"Delays subject to {sup.get('penalty', '2%')} per week penalty.", NM))
    elements.append(Paragraph("3. Short/damaged deliveries to be replaced within 7 working days.", NM))
    elements.append(Paragraph("4. All disputes subject to Delhi jurisdiction only.", NM))

    doc.build(elements)
    return filepath


def main(count: int = 200):
    out = str(Paths.PO_PDFS)
    os.makedirs(out, exist_ok=True)
    print(f"Generating {count} Purchase Order PDFs → {out}")
    for i in range(1, count + 1):
        _build_po_pdf(i, out)
        if i % 50 == 0:
            print(f"  {i}/{count} done")
    print(f"✓ {count} PO PDFs generated.")


if __name__ == "__main__":
    main()
