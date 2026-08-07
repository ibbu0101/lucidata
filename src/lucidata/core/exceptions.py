class LUCIDATAError(Exception):
    """Base exception for all LUCIDATA errors."""


class IngestionError(LUCIDATAError):
    """Raised when data ingestion fails (unsupported format, I/O error, schema mismatch)."""


class LLMProviderError(LUCIDATAError):
    """Raised when an LLM provider call fails irrecoverably (auth, network, invalid response)."""
