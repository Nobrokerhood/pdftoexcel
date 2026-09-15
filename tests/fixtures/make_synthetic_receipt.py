from PIL import Image, ImageDraw, ImageFont
W, H = 900, 1180
im = Image.new("RGB", (W, H), "white"); d = ImageDraw.Draw(im)
F = lambda size, bold=False: ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf", size)
d.rectangle([0, 0, W, 110], fill=(0, 59, 122))
d.text((40, 28), "HDFC BANK", font=F(40, True), fill="white")
d.text((40, 76), "NetBanking - Funds Transfer Acknowledgement", font=F(20), fill="white")
d.text((40, 130), "TEST RECEIPT - SYNTHETIC DATA FOR SOFTWARE TESTING", font=F(18, True), fill=(180, 0, 0))
d.text((40, 175), "Transaction Successful", font=F(30, True), fill=(0, 120, 0))
d.text((40, 225), "Amount Paid", font=F(20), fill=(90, 90, 90))
d.text((40, 252), "\u20b9 12,450.75", font=F(46, True), fill="black")
d.text((40, 312), "(Rupees Twelve Thousand Four Hundred Fifty and Seventy Five Paise Only)", font=F(17), fill=(90, 90, 90))
rows = [
    ("Transaction Date", "03-Sep-2026"),
    ("Transaction Time", "11:42:18 AM"),
    ("Payment Mode", "NEFT"),
    ("UTR / Reference No.", "HDFCN52026090312345678"),
    ("Debit Account", "XXXXXXXX4821 (Savings)"),
    ("Remitter Name", "Test Resident One"),
    ("Beneficiary Name", "Green Meadows Co-op Housing Society"),
    ("Beneficiary Bank", "HDFC Bank Ltd"),
    ("Beneficiary A/c No.", "XXXXXXXX7730"),
    ("IFSC", "HDFC0001234"),
    ("Remarks", "Maintenance charges Jul-Sep 2026"),
    ("", "Tower B Flat 1204"),
]
y = 370
for k, v in rows:
    d.line([40, y - 12, W - 40, y - 12], fill=(220, 220, 220), width=1)
    d.text((40, y), k, font=F(21), fill=(90, 90, 90))
    d.text((360, y), v, font=F(21, True), fill="black")
    y += 62
d.text((40, y + 20), "This is a computer generated acknowledgement and does not require a signature.", font=F(16), fill=(120, 120, 120))
im.save(f"{__import__('os').path.dirname(__file__)}/test_member_receipt.png", optimize=True)
print("saved", im.size)
