import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))


@pytest.fixture
def conn():
    """A migrated, empty database. Set TT_TEST_DATABASE_URL to run DB tests."""
    url = os.environ.get("TT_TEST_DATABASE_URL")
    if not url:
        pytest.skip("TT_TEST_DATABASE_URL not set")
    from tradetracker import db

    c = db.connect(url)
    c.execute("DROP SCHEMA public CASCADE; CREATE SCHEMA public;")
    c.commit()
    db.migrate(c)
    yield c
    c.close()
