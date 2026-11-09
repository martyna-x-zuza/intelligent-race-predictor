"""Small hand-written exports, in the formats of the real Strava and Garmin files."""

from collections.abc import Callable
from pathlib import Path

import pytest

STRAVA_HEADER = (
    "Activity ID,Activity Date,Activity Name,Activity Type,Elapsed Time,Distance,"
    "Max Heart Rate,Moving Time,Average Speed,Average Heart Rate,Max Heart Rate"
)
STRAVA_RUN = '1,"Jul 17, 2021, 3:01:35 PM",Morning Run,Run,3000,10.0,,2900,3.4,150,180'
STRAVA_RIDE = '2,"4 Jul 2017, 07:52:56",Ride,Ride,3600,30.5,170,3500,8.7,140,'

GARMIN_HEADER = (
    "Typ aktywności,Data,Dystans,Czas,Średnie tętno,Maksymalne tętno,Średnie tempo,"
    "Najlepsze tempo,Czas ruchu,Minimalna temperatura,Maksymalna temperatura"
)
GARMIN_RUN = 'Bieganie,2025-09-19 21:33:31,"6,44",00:38:10,152,177,5:56,4:45,00:37:57,20.0,25.0'
GARMIN_RIDE = "Kolarstwo,2025-09-18 17:13:37,30,01:00:00,130,160,30,45,--,--,18.0"
GARMIN_SWIM = "Pływanie na basenie,2025-09-17 07:00:00,1500,00:30:00,--,--,2:00,1:40,--,--,--"

WriteCsv = Callable[[str, str, list[str]], Path]


@pytest.fixture
def write_csv(tmp_path: Path) -> WriteCsv:
    """Write a CSV file with ``header`` and ``rows`` into a temporary directory."""

    def write(name: str, header: str, rows: list[str]) -> Path:
        path = tmp_path / name
        path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
        return path

    return write


@pytest.fixture
def strava_file(write_csv: WriteCsv) -> Path:
    return write_csv("strava.csv", STRAVA_HEADER, [STRAVA_RUN, STRAVA_RIDE])


@pytest.fixture
def garmin_file(write_csv: WriteCsv) -> Path:
    return write_csv("garmin.csv", GARMIN_HEADER, [GARMIN_RUN, GARMIN_RIDE, GARMIN_SWIM])
