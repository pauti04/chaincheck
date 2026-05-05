"""ChainCheck — LLM hallucination detection toolkit."""

from chaincheck.detect import detect
from chaincheck.models import ClaimResult, DetectionResult, MethodResult

__version__ = "0.4.0"
__all__ = ["detect", "DetectionResult", "MethodResult", "ClaimResult", "__version__"]
