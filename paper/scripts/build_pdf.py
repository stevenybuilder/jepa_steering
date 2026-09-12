"""Render four core pages plus references; figures remain separate."""
from pathlib import Path
import html
import json
import re

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
from pypdf import PdfReader
import pymupdf as fitz

PAPER = Path(__file__).resolve().parents[1]
OUT = PAPER / "extended_abstract.pdf"
BLUE = colors.HexColor("#244763")
STYLES = {
    "body":ParagraphStyle("body",fontName="Times-Roman",fontSize=10.8,leading=13.4,
                          spaceAfter=6,alignment=TA_JUSTIFY),
    "title":ParagraphStyle("title",fontName="Times-Bold",fontSize=17,leading=19,
                           spaceAfter=9,textColor=BLUE),
    "section":ParagraphStyle("section",fontName="Times-Bold",fontSize=12,leading=14.2,
                             spaceBefore=8,spaceAfter=6,textColor=BLUE,keepWithNext=True),
    "meta":ParagraphStyle("meta",fontName="Helvetica",fontSize=8.3,leading=10.5,
                          spaceAfter=9,textColor=colors.HexColor("#596674")),
    "bullet":ParagraphStyle("bullet",fontName="Times-Roman",fontSize=10.8,leading=13.4,
                            leftIndent=9,firstLineIndent=-9,spaceAfter=6),
    "cell":ParagraphStyle("cell",fontName="Helvetica",fontSize=8.1,leading=10,spaceAfter=0),
    "ref":ParagraphStyle("ref",fontName="Times-Roman",fontSize=10.5,leading=13.5,spaceAfter=10),
}
URLS={1:"https://arxiv.org/abs/2603.03276",2:"https://arxiv.org/abs/2512.24497v4",
      3:"https://arxiv.org/abs/2602.07050",4:"https://arxiv.org/abs/2605.17144",
      5:"https://arxiv.org/abs/2605.05115",6:"https://arxiv.org/abs/2608.12939",
      7:"https://arxiv.org/abs/2603.19312v3",8:"https://arxiv.org/abs/2608.27395v1",
      9:"https://arxiv.org/abs/2606.26217",10:"https://arxiv.org/abs/2607.05238v3",
      11:"https://arxiv.org/abs/2404.03592",12:"https://arxiv.org/abs/2309.16042v2"}

def inline(text):
    text=text.replace("\u2013","-").replace("\u2014"," - ").replace("\u2011","-")
    text=text.replace("\u2019","'").replace("\u2018","'").replace("\u201c",'"').replace("\u201d",'"')
    text=html.escape(text)
    text=re.sub(r"\*\*(.+?)\*\*",r"<b>\1</b>",text)
    text=re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)",r"<i>\1</i>",text)
    text=re.sub(r"`([^`]+)`",r'<font name="Courier">\1</font>',text)
    text=re.sub(r"\[([^]]+)\]\((https?://[^)]+)\)",r'<link href="\2" color="#245f96">\1</link>',text)
    def refs(m):
        return "["+", ".join(f'<link href="{URLS[int(n)]}" color="#245f96">{n}</link>' for n in m[1].split(","))+"]"
    text=re.sub(r"\[(\d+(?:,\d+)*)\]",refs,text)
    return text

def parse(md):
    lines=md.splitlines(); result=[]; i=0
    while i<len(lines):
        line=lines[i].strip()
        if not line: i+=1;continue
        if line.startswith("|"):
            raw=[]
            while i<len(lines) and lines[i].strip().startswith("|"):
                cells=[x.strip() for x in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r"[:\- ]+",x) for x in cells):raw.append(cells)
                i+=1
            data=[[Paragraph(inline(("**"+x+"**") if r==0 else x),STYLES["cell"]) for x in row] for r,row in enumerate(raw)]
            t=Table(data,colWidths=[102,238,172],repeatRows=1,hAlign="LEFT")
            t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#eaf0f5")),
                                  ("VALIGN",(0,0),(-1,-1),"TOP"),
                                  ("LINEBELOW",(0,0),(-1,0),.7,BLUE),
                                  ("LINEBELOW",(0,-1),(-1,-1),.6,BLUE),
                                  ("TOPPADDING",(0,0),(-1,-1),5),
                                  ("BOTTOMPADDING",(0,0),(-1,-1),5)]))
            result.extend([t,Spacer(1,8)]);continue
        if line.startswith("# "):key,text="title",line[2:]
        elif line.startswith("## "):key,text="section",line[3:]
        elif line.startswith("- "):key,text="bullet","- "+line[2:]
        elif line.startswith("*Development"):key,text="meta",line.strip("*")
        else:key,text="body",line
        result.append(Paragraph(inline(text),STYLES[key]));i+=1
    return result

md=(PAPER/"extended_abstract.md").read_text()
pages=md.split("<!-- pagebreak -->")
assert len(pages)==4
story=[]
for i,page in enumerate(pages):
    if i:story.append(PageBreak())
    story.extend(parse(page))
story.append(PageBreak())
story.append(Paragraph("References",STYLES["title"]))
for line in (PAPER/"references.md").read_text().splitlines():
    if re.match(r"^\d+\. ",line):story.append(Paragraph(inline(line),STYLES["ref"]))
story.append(Spacer(1,14))
story.append(Paragraph("Figures, evidence, and reproducibility",STYLES["section"]))
story.append(Paragraph("Six figures are supplied separately in the paper/figures directory. "
                       "The paper directory also contains editable Markdown, the methodology-alignment "
                       "table, task coverage, an evidence appendix, source-bound contrast data, and "
                       "rebuild scripts. All plotted results are completed offline development evidence. "
                       "No partial behavioral outcome is included.",STYLES["body"]))

def decoration(c,doc):
    c.setFont("Helvetica",8)
    c.setFillColor(colors.HexColor("#6c7781"))
    c.drawString(50,768,"DEVELOPMENT RESULTS | 8 SEPTEMBER 2026")
    c.drawRightString(562,768,"FROZEN JEPA WORLD MODELS")
    c.setStrokeColor(colors.HexColor("#d1d9df"));c.line(50,758,562,758)
    c.drawCentredString(306,25,f"{doc.page}" if doc.page<=4 else "References - excluded from four-page core")

doc=SimpleDocTemplate(str(OUT),pagesize=letter,leftMargin=50,rightMargin=50,
                      topMargin=45,bottomMargin=43,title="Correcting Imagined Futures in Frozen JEPA World Models",
                      author="Development research draft",pageCompression=1)
doc.build(story,onFirstPage=decoration,onLaterPages=decoration)
reader=PdfReader(OUT)
assert 5 <= len(reader.pages) <= 6, f"Unexpected length: {len(reader.pages)} pages"
expected=["1. Introduction","2. Study design","3. Completed offline evidence","4. Interpretation","References"]
for page,term in zip(reader.pages,expected):assert term in page.extract_text(),term
qa=PAPER/"review"
qa.mkdir(exist_ok=True)
pdf=fitz.open(OUT)
for i in range(len(pdf)):
    pdf[i].get_pixmap(matrix=fitz.Matrix(1.25,1.25)).save(qa/f"page-{i+1:02d}.png")
checks={"core_pages":4,"reference_pages":len(reader.pages)-4,"total_pages":len(reader.pages),
        "core_words_by_page":[len(re.findall(r"\b[\w'-]+\b",p)) for p in pages],
        "expected_sections_present":True,"pdf_path":"extended_abstract.pdf",
        "visual_review":"Renderings saved; inspect before delivery."}
(qa/"layout_check.json").write_text(json.dumps(checks,indent=2)+"\n")
print(json.dumps(checks))
