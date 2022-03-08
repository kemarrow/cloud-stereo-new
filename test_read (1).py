import gzip
import typing
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset, chartostring

test_file = Path("20211024_1200.gz")
airports = np.array(["EGLL", "EGSS", "EGKK", "EGMC", "EGGW", "EGMC"])


def test_read(
    filename: Path,
    extract_keys: typing.List[str] = [
        "profileAirport",
        "profileTime",
        "altitude",
        "windDir",
        "windSpeed",
    ],
) -> typing.Dict[str, np.ndarray]:
    output = {}
    with gzip.open(filename) as gz:
        with Dataset("dummy", mode="r", memory=gz.read()) as ncdf:
            print(ncdf.variables.keys())
            for key in extract_keys:
                tmp = ncdf.variables[key][:]
                if tmp.dtype.str.startswith("|S"):
                    tmp = chartostring(tmp)
                output[key] = tmp
    return output


data = test_read(test_file)

airport_idx = np.array([code in airports for code in data["profileAirport"]])
print(airport_idx.sum())
data_airports = {key: value[airport_idx] for key, value in data.items()}

for airport, alt, speed in zip(
    data_airports["profileAirport"],
    data_airports["altitude"],
    data_airports["windSpeed"],
):
    plt.plot(speed, alt)
plt.show()
