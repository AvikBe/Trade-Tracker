"""Builds a tiny quarterly data set ZIP in the SEC's layout for tests."""

import zipfile
from pathlib import Path

SUBMISSION = [
    ["ACCESSION_NUMBER", "FILING_DATE", "PERIOD_OF_REPORT", "DOCUMENT_TYPE", "ISSUERCIK",
     "ISSUERNAME", "ISSUERTRADINGSYMBOL", "REMARKS", "AFF10B5ONE"],
    ["0000000001-24-000001", "06-MAR-2024", "04-MAR-2024", "4", "0001234567",
     "Example Widgets Inc", "EXWD", "", "0"],
    ["0000000001-24-000002", "20-MAR-2024", "04-MAR-2024", "4/A", "0001234567",
     "Example Widgets Inc", "EXWD", "", "0"],
    ["0000000001-24-000003", "07-MAR-2024", "05-MAR-2024", "4", "0000999999",
     "Other Co", "OTHR", "Sales under a 10b5-1 plan.", "0"],
    ["0000000001-24-000004", "07-MAR-2024", "05-MAR-2024", "3", "0000999999",
     "Other Co", "OTHR", "", "0"],
]
REPORTINGOWNER = [
    ["ACCESSION_NUMBER", "RPTOWNERCIK", "RPTOWNERNAME", "RPTOWNER_RELATIONSHIP", "RPTOWNER_TITLE"],
    ["0000000001-24-000001", "0007654321", "Doe Jane", "Director,Officer", "CFO"],
    ["0000000001-24-000002", "0007654321", "Doe Jane", "Director,Officer", "CFO"],
    ["0000000001-24-000003", "0000111111", "Roe Rich", "TenPercentOwner", ""],
]
NONDERIV_TRANS = [
    ["ACCESSION_NUMBER", "NONDERIV_TRANS_SK", "SECURITY_TITLE", "TRANS_DATE", "TRANS_CODE",
     "TRANS_SHARES", "TRANS_PRICEPERSHARE", "TRANS_PRICEPERSHARE_FN",
     "DIRECT_INDIRECT_OWNERSHIP", "NATURE_OF_OWNERSHIP"],
    ["0000000001-24-000001", "11", "Common Stock", "04-MAR-2024", "P", "10000", "12.34", "F1", "D", ""],
    ["0000000001-24-000001", "12", "Common Stock", "04-MAR-2024", "F", "300", "12.30", "", "D", ""],
    ["0000000001-24-000002", "21", "Common Stock", "04-MAR-2024", "P", "10000", "12.34", "", "D", ""],
    ["0000000001-24-000003", "31", "Common Stock", "05-MAR-2024", "S", "5000", "40", "", "I", "By Trust"],
]
FOOTNOTES = [
    ["ACCESSION_NUMBER", "FOOTNOTE_ID", "FOOTNOTE_TXT"],
    ["0000000001-24-000001", "F1", "Weighted average price."],
]


def write(path: Path) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        for name, rows in [("SUBMISSION.tsv", SUBMISSION), ("REPORTINGOWNER.tsv", REPORTINGOWNER),
                           ("NONDERIV_TRANS.tsv", NONDERIV_TRANS), ("FOOTNOTES.tsv", FOOTNOTES)]:
            zf.writestr(name, "\n".join("\t".join(r) for r in rows) + "\n")
    return path
