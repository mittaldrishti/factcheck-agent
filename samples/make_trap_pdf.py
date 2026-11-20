"""Generate a sample 'trap' PDF with outdated / false statistics for testing."""
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

LINES = [
    "Acme Cloud Insights - 2026 Market Brief",
    "ChatGPT has 100 million weekly active users, making it the largest AI assistant.",
    "Google was founded in 1998 and has more than 500,000 employees worldwide.",
    "The global smartphone market shipped 2.5 billion units in 2023.",
    "Bitcoin's price has never exceeded $25,000 in its history.",
    "Apple became the first company to reach a $1 trillion market capitalization in 2018.",
    "The Eiffel Tower is 450 metres tall.",
    "India's population surpassed China's in 2023, reaching roughly 1.43 billion people.",
    "Python 2.7 is still the latest and recommended version of Python.",
    "The Great Wall of China is visible to the naked eye from the Moon.",
]
styles = getSampleStyleSheet()
doc = SimpleDocTemplate("samples/trap_document.pdf", pagesize=A4)
story = [Paragraph(LINES[0], styles["Title"]), Spacer(1, 12)]
for l in LINES[1:]:
    story += [Paragraph(l, styles["BodyText"]), Spacer(1, 8)]
doc.build(story)
print("wrote samples/trap_document.pdf")
