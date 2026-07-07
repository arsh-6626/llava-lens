"""Global registry for models, analyzers, and other components."""

from typing import Any, Callable, Dict, List, Optional, Type
import logging

logger = logging.getLogger(__name__)


class Registry:
    """Singleton registry for all pluggable components."""

    _instance: Optional["Registry"] = None
    _registries: Dict[str, Dict[str, Any]] = {}

    def __new__(cls) -> "Registry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._registries = {}
        return cls._instance

    def register(self, category: str, name: str, component: Any) -> None:
        """Register a component under a category."""
        if category not in self._registries:
            self._registries[category] = {}
        self._registries[category][name] = component
        logger.debug(f"Registered {name} in {category}")

    def get(self, category: str, name: str) -> Any:
        """Get a registered component."""
        if category not in self._registries:
            raise KeyError(f"Category '{category}' not found in registry")
        if name not in self._registries[category]:
            available = list(self._registries[category].keys())
            raise KeyError(f"'{name}' not found in {category}. Available: {available}")
        return self._registries[category][name]

    def list_available(self, category: str) -> List[str]:
        """List available components in a category."""
        return list(self._registries.get(category, {}).keys())

    def list_categories(self) -> List[str]:
        """List all registered categories."""
        return list(self._registries.keys())

    def register_model(self, name: str, cls: Type) -> Type:
        """Decorator to register a model class."""
        self.register("models", name, cls)
        return cls

    def register_analyzer(self, name: str, cls: Type) -> Type:
        """Decorator to register an analyzer class."""
        self.register("analyzers", name, cls)
        return cls

    def clear(self) -> None:
        """Clear all registrations."""
        self._registries.clear()


def get_registry() -> Registry:
    """Get the global registry instance."""
    return Registry()
