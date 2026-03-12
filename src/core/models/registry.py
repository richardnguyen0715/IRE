"""Model registry for managing detection model implementations.

Provides a centralized registration and lookup mechanism for model
classes. New model architectures register themselves via the
``@ModelRegistry.register("name")`` decorator and can be instantiated
dynamically from configuration files.
"""

from typing import Any, Dict, List, Type

from src.core.models.base import BaseDetector


class ModelRegistry:
    """Registry for detection model implementations.

    A class-level registry that maps model architecture names to their
    implementation classes. Supports decorator-based registration and
    factory-style instantiation.

    Example::

        @ModelRegistry.register("my_model")
        class MyDetector(BaseDetector):
            ...

        # Later, create an instance dynamically
        detector = ModelRegistry.create("my_model", config)

    Class Attributes:
        _models: Internal dictionary mapping names to model classes.
    """

    _models: Dict[str, Type[BaseDetector]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a model class under a given name.

        Args:
            name: Unique identifier for the model architecture.

        Returns:
            Decorator function that registers the class and returns
            it unchanged.

        Raises:
            ValueError: If a model with the same name is already
                        registered.
        """

        def decorator(model_cls: Type[BaseDetector]):
            if name in cls._models:
                raise ValueError(
                    f"Model '{name}' is already registered by "
                    f"{cls._models[name].__name__}"
                )
            cls._models[name] = model_cls
            return model_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> Type[BaseDetector]:
        """Retrieve a registered model class by name.

        Args:
            name: Name of the registered model architecture.

        Returns:
            The model class associated with the given name.

        Raises:
            ValueError: If no model is registered with the given name.
        """
        if name not in cls._models:
            available = ", ".join(cls._models.keys()) or "(none)"
            raise ValueError(
                f"Model '{name}' not found. "
                f"Available models: {available}"
            )
        return cls._models[name]

    @classmethod
    def list_models(cls) -> List[str]:
        """List all registered model architecture names.

        Returns:
            Sorted list of registered model name strings.
        """
        return sorted(cls._models.keys())

    @classmethod
    def create(cls, name: str, config: Dict[str, Any]) -> BaseDetector:
        """Create a model instance by architecture name and config.

        Factory method that looks up the model class in the registry
        and instantiates it with the provided configuration.

        Args:
            name: Name of the registered model architecture.
            config: Configuration dictionary for the model.

        Returns:
            Instantiated detector model.
        """
        model_cls = cls.get(name)
        return model_cls(config)
