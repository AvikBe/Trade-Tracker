"""Settings read from the environment. Secrets never live in the repo."""

import os
from dataclasses import dataclass


class ConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class Settings:
    database_url: str
    edgar_user_agent: str | None
    tiingo_api_key: str | None

    def require_edgar(self) -> str:
        ua = self.edgar_user_agent
        if not ua or "@" not in ua:
            raise ConfigError(
                "EDGAR_USER_AGENT must be set to a name plus contact email, "
                "e.g. 'Trade Tracker you@example.com' (SEC fair-access policy)."
            )
        return ua

    def require_tiingo(self) -> str:
        if not self.tiingo_api_key:
            raise ConfigError("TIINGO_API_KEY is not set.")
        return self.tiingo_api_key


def load() -> Settings:
    return Settings(
        database_url=os.environ.get(
            "DATABASE_URL", "postgresql://tt:tt@localhost:5432/tradetracker"
        ),
        edgar_user_agent=os.environ.get("EDGAR_USER_AGENT"),
        tiingo_api_key=os.environ.get("TIINGO_API_KEY"),
    )
