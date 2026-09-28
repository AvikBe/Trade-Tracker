from tradetracker.edgar.feed import parse_feed

ATOM = b"""<?xml version="1.0" encoding="ISO-8859-1" ?>
<feed xmlns="http://www.w3.org/2005/Atom">
<title>Latest Filings</title>
<entry>
<title>4 - Doe Jane (0007654321) (Reporting)</title>
<link rel="alternate" type="text/html" href="https://www.sec.gov/Archives/edgar/data/7654321/000123456724000001/0001234567-24-000001-index.htm"/>
<summary type="html"> &lt;b&gt;Filed:&lt;/b&gt; 2024-03-06</summary>
<updated>2024-03-06T16:31:05-05:00</updated>
<category scheme="https://www.sec.gov/" label="form type" term="4"/>
<id>urn:tag:sec.gov,2008:accession-number=0001234567-24-000001</id>
</entry>
</feed>"""


def test_parse_feed():
    [e] = parse_feed(ATOM)
    assert e.accession == "0001234567-24-000001"
    assert e.form_type == "4"
    assert e.cik == "7654321"
    assert e.accepted_at.isoformat() == "2024-03-06T16:31:05-05:00"


def _entry(acc, cik, role, form="4"):
    return f"""<entry>
<title>{form} - X ({cik}) ({role})</title>
<link rel="alternate" type="text/html" href="https://www.sec.gov/Archives/edgar/data/{cik}/{acc.replace('-', '')}/{acc}-index.htm"/>
<updated>2024-03-06T16:31:05-05:00</updated>
<category scheme="https://www.sec.gov/" label="form type" term="{form}"/>
<id>urn:tag:sec.gov,2008:accession-number={acc}</id>
</entry>"""


def test_fetch_latest_dedupes_parties_and_keeps_only_form_4():
    import httpx

    from tradetracker.edgar.client import EdgarClient
    from tradetracker.edgar.feed import fetch_latest

    body = (
        '<feed xmlns="http://www.w3.org/2005/Atom">'
        + _entry("0000000001-24-000001", "111", "Issuer")
        + _entry("0000000001-24-000001", "222", "Reporting")
        + _entry("0000000001-24-000002", "111", "Issuer", form="4/A")
        + _entry("0000000001-24-000003", "111", "Issuer", form="3")
        + "</feed>"
    ).encode()
    c = EdgarClient("t t@x.com", transport=httpx.MockTransport(lambda r: httpx.Response(200, content=body)))
    entries = fetch_latest(c)
    assert [(e.accession, e.form_type) for e in entries] == [
        ("0000000001-24-000001", "4"), ("0000000001-24-000002", "4/A")
    ]


def test_ownership_xml_url_skips_rendered_copies():
    import httpx

    from tradetracker.edgar.client import EdgarClient
    from tradetracker.edgar.feed import ownership_xml_url

    [e] = parse_feed(ATOM)
    listing = {"directory": {"item": [
        {"name": "0001234567-24-000001-index.htm"},
        {"name": "xslF345X05"},
        {"name": "xslF345X05_wk-form4.xml"},
        {"name": "wk-form4_1709760665.xml"},
    ]}}
    c = EdgarClient("t t@x.com", transport=httpx.MockTransport(lambda r: httpx.Response(200, json=listing)))
    assert ownership_xml_url(c, e).endswith("/7654321/000123456724000001/wk-form4_1709760665.xml")
