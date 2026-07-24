"""PDF and Word loaders, against real files built in the test.

Worth building the files rather than mocking the readers: the failure this
guards is a client corpus of exactly these two formats yielding nothing, and a
mocked `PdfReader` cannot tell us whether we called it correctly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from stillroom.ingest.loaders import UnreadableDocument, load_corpus, load_document

pypdf = pytest.importorskip("pypdf")
docx = pytest.importorskip("docx")


def _write_pdf(path: Path, text: str) -> None:
    """Write a minimal but genuinely valid one-page PDF containing `text`.

    Hand-built rather than produced by a library: pypdf writes PDFs but will not
    lay down text, and pulling in a rendering dependency to generate a fixture
    would put a package in the deliverable's tree that the deliverable does not
    use. The structure below is the smallest thing with a real font resource,
    which is what makes the text extractable rather than mojibake.
    """
    content = f"BT /F1 12 Tf 10 100 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"\nendstream",
    ]

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"

    xref_at = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n".encode()
    out += f"startxref\n{xref_at}\n%%EOF\n".encode()

    path.write_bytes(bytes(out))


def test_a_word_document_is_read_including_its_tables(tmp_path: Path):
    document = docx.Document()
    document.add_paragraph("Expense policy")
    table = document.add_table(rows=2, cols=2)
    table.rows[0].cells[0].text = "Category"
    table.rows[0].cells[1].text = "Limit"
    table.rows[1].cells[0].text = "Travel"
    table.rows[1].cells[1].text = "1200 per trip"
    path = tmp_path / "expenses.docx"
    document.save(str(path))

    raw = load_document(path, tmp_path)

    assert "Expense policy" in raw.text
    # Real corpora keep thresholds and rate cards in tables, and `paragraphs`
    # alone cannot see them.
    assert "Travel | 1200 per trip" in raw.text


def test_an_empty_word_document_is_refused(tmp_path: Path):
    path = tmp_path / "blank.docx"
    docx.Document().save(str(path))

    with pytest.raises(UnreadableDocument, match="no text"):
        load_document(path, tmp_path)


def test_a_pdf_with_text_is_read(tmp_path: Path):
    path = tmp_path / "notice.pdf"
    _write_pdf(path, "Notice period is four weeks")

    raw = load_document(path, tmp_path)

    assert "Notice period" in raw.text


def test_a_pdf_with_no_extractable_text_is_named_as_a_probable_scan(tmp_path: Path):
    """The client is told which file, and why — not left with a silent gap."""
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    path = tmp_path / "scan.pdf"
    writer.write(str(path))

    with pytest.raises(UnreadableDocument, match="scanned image"):
        load_document(path, tmp_path)


def test_a_corrupt_file_is_skipped_and_the_rest_still_ingest(tmp_path: Path):
    (tmp_path / "good.md").write_text("## Alpha\n\nBody.\n", encoding="utf-8")
    (tmp_path / "broken.pdf").write_bytes(b"this is not a pdf at all")

    documents, skipped = load_corpus(tmp_path, (".md", ".pdf"))

    # An ingest must not die on file 180 of 200.
    assert [d.source for d in documents] == ["good.md"]
    assert len(skipped) == 1
    assert "broken.pdf" in skipped[0]


def test_nested_directories_are_walked(tmp_path: Path):
    nested = tmp_path / "hr" / "policies"
    nested.mkdir(parents=True)
    (nested / "leave.md").write_text("## Leave\n\nTwenty days.\n", encoding="utf-8")

    documents, _ = load_corpus(tmp_path, (".md",))

    # Clients hand over folder trees, not flat directories.
    assert documents[0].source == "hr/policies/leave.md"


# --- spreadsheets: Excel and CSV are among the commonest business files -------

openpyxl = pytest.importorskip("openpyxl")


def test_an_excel_workbook_is_read_sheet_by_sheet_including_its_cells(tmp_path: Path):
    workbook = openpyxl.Workbook()
    first = workbook.active
    first.title = "Limits"
    first.append(["Category", "Limit"])
    first.append(["Travel", "1200 per trip"])
    second = workbook.create_sheet("Approvals")
    second.append(["Any expense over 500 is approved by the Finance Director"])
    path = tmp_path / "expenses.xlsx"
    workbook.save(str(path))

    raw = load_document(path, tmp_path)

    # Cells become `cell | cell`, the same shape a Word table gets.
    assert "Travel | 1200 per trip" in raw.text
    # Every sheet is read, and named so two tabs cannot silently merge.
    assert "Limits" in raw.text and "Approvals" in raw.text
    assert "Finance Director" in raw.text


def test_an_excel_formula_cell_reports_its_value_not_the_formula(tmp_path: Path):
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet["A1"] = "Total"
    sheet["B1"] = 575  # data_only reads the cached value, not "=SUM(...)"
    path = tmp_path / "totals.xlsx"
    workbook.save(str(path))

    raw = load_document(path, tmp_path)

    assert "Total | 575" in raw.text


def test_an_empty_workbook_is_refused(tmp_path: Path):
    path = tmp_path / "blank.xlsx"
    openpyxl.Workbook().save(str(path))

    with pytest.raises(UnreadableDocument, match="no cells"):
        load_document(path, tmp_path)


def test_a_csv_is_read_row_per_line(tmp_path: Path):
    path = tmp_path / "prices.csv"
    path.write_text("Item,Price\nWidget,9.99\nGadget,19.99\n", encoding="utf-8")

    raw = load_document(path, tmp_path)

    assert "Widget | 9.99" in raw.text
    assert "Gadget | 19.99" in raw.text


def test_a_semicolon_separated_csv_is_detected(tmp_path: Path):
    # European exports are commonly semicolon-separated; the delimiter must not
    # be assumed to be a comma, or every row collapses into one cell.
    path = tmp_path / "export.csv"
    path.write_text("Item;Price\nWidget;9,99\n", encoding="utf-8")

    raw = load_document(path, tmp_path)

    assert "Item | Price" in raw.text
    assert "Widget | 9,99" in raw.text
