import wfdb  # pyright: ignore[reportMissingTypeStubs]
import numpy as np
from glob import glob
from numpy.typing import NDArray
from typing import Any
from tqdm import tqdm

result: dict[str, NDArray[Any]] = {}
for type in ["N", "A", "V", "L", "R"]:
    result[type] = np.empty([0, 300], dtype=np.float32)


def is_target(symbol: str) -> bool:
    return symbol in ["N", "A", "V", "L", "R"]


for path in tqdm(glob("./assets/*.dat", recursive=False)):
    record_name: str = path[:-4]
    record = np.array(
        wfdb.rdrecord(record_name, channel_names=["MLII"]).p_signal,
        dtype=np.float32,
    )
    annotation = wfdb.rdann(record_name, "atr")
    if annotation.symbol is None:
        continue
    for index, symbol in enumerate(annotation.symbol):  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
        assert isinstance(symbol, str)
        if not is_target(symbol):
            continue
        try:
            # size limit
            if result[symbol].shape[0] >= 1000:
                continue
            # 60% chance to skip
            if np.random.rand() >= 0.4:
                continue
            # no cross-boundary
            if (
                annotation.sample[index] < 99
                or annotation.sample[index] > record.shape[0] - 201
            ):
                continue
            result[symbol] = np.vstack(
                (
                    result[symbol],
                    record[
                        annotation.sample[index] - 99 : annotation.sample[index] + 201
                    ].transpose(),
                )
            )
        except IndexError:
            continue


for key in result.keys():
    print(key, result[key].shape)
    np.save(f"./{key}.npy", result[key])
