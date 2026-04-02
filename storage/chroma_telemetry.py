"""No-op telemetry adapter for Chroma in local deployments."""

from chromadb.telemetry.product import ProductTelemetryClient
from chromadb.telemetry.product.events import ProductTelemetryEvent
from overrides import override


class NoOpProductTelemetryClient(ProductTelemetryClient):
    """Drop telemetry events to keep local/offline runs quiet."""

    @override
    def capture(self, event: ProductTelemetryEvent) -> None:  # noqa: ARG002
        return
