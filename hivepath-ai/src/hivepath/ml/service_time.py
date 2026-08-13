"""Per-stop service time prediction.

Two implementations share one interface. The heuristic always works and needs no
dependencies; the neural model is used when torch and a trained checkpoint are
both present.

Three things are deliberately different from the previous implementation:

* Models load **lazily**, on first prediction, not at import. Importing a module
  should not read 50MB of checkpoints or emit output.
* Artefact paths resolve against configuration, not the process working
  directory, so running from ``src/`` and from the repository root behave alike.
* Accessibility is converted from the domain's 0-100 scale to the 0-1 scale the
  models were trained on, in exactly one place.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from datetime import datetime
from functools import lru_cache

from hivepath.config import Settings, get_settings
from hivepath.domain import Stop
from hivepath.logging_config import get_logger

logger = get_logger(__name__)

#: Never predict a stop shorter than this; a delivery has irreducible overhead.
MIN_SERVICE_MINUTES = 3.0
MAX_SERVICE_MINUTES = 120.0


class ServiceTimeModel(ABC):
    """Predicts how many minutes a stop will take."""

    name: str = "abstract"

    @abstractmethod
    def predict(self, stops: Sequence[Stop]) -> list[float]:
        """Predicted minutes, positionally aligned with ``stops``."""

    def predict_by_id(self, stops: Sequence[Stop]) -> dict[str, float]:
        """Predictions keyed by stop id, which is what callers usually want."""
        return dict(zip((s.id for s in stops), self.predict(stops), strict=True))


class HeuristicServiceTimeModel(ServiceTimeModel):
    """Closed-form fallback: base cost, plus load, plus an accessibility surcharge.

    Used whenever no trained checkpoint is available. The coefficients come from
    the original implementation and are retained so behaviour is unchanged where
    the neural model is absent.
    """

    name = "heuristic"

    BASE_MINUTES = 4.0
    MINUTES_PER_DEMAND_UNIT = 0.06
    MAX_ACCESS_SURCHARGE_MIN = 5.0

    def predict(self, stops: Sequence[Stop]) -> list[float]:
        predictions = []
        for stop in stops:
            minutes = (
                self.BASE_MINUTES
                + self.MINUTES_PER_DEMAND_UNIT * stop.demand
                + self.MAX_ACCESS_SURCHARGE_MIN * (1.0 - stop.access_fraction)
            )
            predictions.append(_clamp_minutes(minutes))
        return predictions


class NeuralServiceTimeModel(ServiceTimeModel):
    """Torch MLP over (demand, accessibility, hour, weekday)."""

    name = "neural"

    def __init__(self, checkpoint_path, torch_module) -> None:
        self._torch = torch_module
        checkpoint = torch_module.load(checkpoint_path, map_location="cpu", weights_only=False)

        nn = torch_module.nn
        model = nn.Sequential(
            nn.Linear(6, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        state = checkpoint.get("state_dict", checkpoint)
        # The checkpoint stores the MLP under an "mlp." prefix.
        model.load_state_dict(
            {k.removeprefix("mlp."): v for k, v in state.items()}, strict=True
        )
        model.eval()
        self._model = model

    def predict(self, stops: Sequence[Stop]) -> list[float]:
        now = datetime.now()
        features = [
            [
                float(stop.demand),
                stop.access_fraction,
                float(stop.demand),
                stop.access_fraction,
                float(now.hour),
                float(now.weekday()),
            ]
            for stop in stops
        ]
        tensor = self._torch.tensor(features, dtype=self._torch.float32)
        with self._torch.no_grad():
            raw = self._model(tensor).squeeze(-1).tolist()
        if isinstance(raw, float):
            raw = [raw]
        return [_clamp_minutes(value) for value in raw]


def _clamp_minutes(value: float) -> float:
    return float(min(MAX_SERVICE_MINUTES, max(MIN_SERVICE_MINUTES, value)))


def _load_neural_model(settings: Settings) -> ServiceTimeModel | None:
    """Attempt to build the neural model; ``None`` if unavailable."""
    checkpoint = settings.artifacts_dir / "service_time_mlp.pt"
    if not checkpoint.is_file():
        logger.debug("no service time checkpoint at %s", checkpoint)
        return None

    try:
        import torch
    except ImportError:
        logger.info(
            "torch not installed; using heuristic service times "
            "(install the 'ml' extra to enable the neural model)"
        )
        return None

    try:
        model = NeuralServiceTimeModel(checkpoint, torch)
    except Exception:
        logger.warning(
            "service time checkpoint at %s could not be loaded; using heuristic",
            checkpoint,
            exc_info=True,
        )
        return None

    logger.info("loaded neural service time model from %s", checkpoint)
    return model


@lru_cache(maxsize=1)
def get_service_time_model() -> ServiceTimeModel:
    """Best available model, resolved once per process.

    Call ``get_service_time_model.cache_clear()`` in tests to force reselection.
    """
    settings = get_settings()
    return _load_neural_model(settings) or HeuristicServiceTimeModel()


def predict_service_minutes(stops: Sequence[Stop]) -> dict[str, float]:
    """Predict service minutes for stops, keyed by stop id."""
    if not stops:
        return {}
    return get_service_time_model().predict_by_id(stops)


def apply_service_times(stops: Sequence[Stop], *, overwrite: bool = False) -> list[Stop]:
    """Set ``Stop.service_min`` in place for stops that lack an explicit value."""
    stops = list(stops)
    pending = [s for s in stops if overwrite or s.service_min is None]
    if not pending:
        return stops

    predictions = predict_service_minutes(pending)
    for stop in pending:
        predicted = predictions.get(stop.id)
        if predicted is not None:
            stop.service_min = max(1, round(predicted))
    return stops
