from dataclasses import dataclass


@dataclass
class DataModel:
    time: str
    temperature_2m: float
    relativehumidity_2m: float
    dewpoint_2m: float
    windspeed_10m: float
    windspeed_100m: float
    winddirection_10m: float
    winddirection_100m: float
    windgusts_10m: float
    power: float

    def to_feature_list(self) -> list[float]:
        return [
            self.temperature_2m,
            self.relativehumidity_2m,
            self.dewpoint_2m,
            self.windspeed_10m,
            self.windspeed_100m,
            self.winddirection_10m,
            self.winddirection_100m,
            self.windgusts_10m,
        ]


@dataclass
class ClientModel:
    id: int
    data: list[DataModel]

    def __init__(self, id: int = 0):
        self.id = id
        self.data = []
