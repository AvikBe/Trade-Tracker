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
