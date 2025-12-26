import json
from typing import Any, Dict, Iterable, List

from torchvision import transforms

from .base import BaseMapper
from .mappers_config import (
    KeyRenameMapperConfig,
    LightParamsMapperConfig,
    RescaleMapperConfig,
    TorchvisionMapperConfig,
)


class KeyRenameMapper(BaseMapper):
    """
    Rename keys in a sample according to a key map

    Args:

        config (KeyRenameMapperConfig): Configuration for the mapper

    Examples
    ########

    1. Rename keys in a sample according to a key map

    .. code-block:: python

        from cr.data.mappers import KeyRenameMapper, KeyRenameMapperConfig

        config = KeyRenameMapperConfig(
            key_map={"old_key": "new_key"}
        )

        mapper = KeyRenameMapper(config)

        sample = {"old_key": 1}
        new_sample = mapper(sample)
        print(new_sample)  # {"new_key": 1}

    2. Rename keys in a sample according to a key map and a condition key

    .. code-block:: python

        from cr.data.mappers import KeyRenameMapper, KeyRenameMapperConfig

        config = KeyRenameMapperConfig(
            key_map={"old_key": "new_key"},
            condition_key="condition",
            condition_fn=lambda x: x == 1
        )

        mapper = KeyRenameMapper(config)

        sample = {"old_key": 1, "condition": 1}
        new_sample = mapper(sample)
        print(new_sample)  # {"new_key": 1}

        sample = {"old_key": 1, "condition": 0}
        new_sample = mapper(sample)
        print(new_sample)  # {"old_key": 1}

    ```
    """

    def __init__(self, config: KeyRenameMapperConfig):
        super().__init__(config)
        self.key_map = config.key_map
        self.condition_key = config.condition_key
        self.condition_fn = config.condition_fn
        self.else_key_map = config.else_key_map

    def __call__(self, batch: Dict[str, Any], *args, **kwrags):
        if self.condition_key is not None:
            condition_key = batch[self.condition_key]
            if self.condition_fn(condition_key):
                for old_key, new_key in self.key_map.items():
                    if old_key in batch:
                        batch[new_key] = batch.pop(old_key)

            elif self.else_key_map is not None:
                for old_key, new_key in self.else_key_map.items():
                    if old_key in batch:
                        batch[new_key] = batch.pop(old_key)

        else:
            for old_key, new_key in self.key_map.items():
                if old_key in batch:
                    batch[new_key] = batch.pop(old_key)
        return batch


class TorchvisionMapper(BaseMapper):
    """
    Apply torchvision transforms to a sample

    Args:

        config (TorchvisionMapperConfig): Configuration for the mapper
    """

    def __init__(self, config: TorchvisionMapperConfig):
        super().__init__(config)
        chained_transforms = []
        for transform, kwargs in zip(config.transforms, config.transforms_kwargs):
            transform = getattr(transforms, transform)
            chained_transforms.append(transform(**kwargs))
        self.transforms = transforms.Compose(chained_transforms)

    def __call__(self, batch: Dict[str, Any], *args, **kwrags) -> Dict[str, Any]:
        if self.key in batch:
            batch[self.output_key] = self.transforms(batch[self.key])
        return batch


class RescaleMapper(BaseMapper):
    """
    Rescale a sample from [0, 1] to [-1, 1]

    Args:

        config (RescaleMapperConfig): Configuration for the mapper
    """

    def __init__(self, config: RescaleMapperConfig):
        super().__init__(config)

    def __call__(self, batch: Dict[str, Any], *args, **kwrags) -> Dict[str, Any]:
        if isinstance(batch[self.key], list):
            tmp = []
            for i, image in enumerate(batch[self.key]):
                tmp.append(2 * image - 1)
            batch[self.output_key] = tmp
        else:
            batch[self.output_key] = 2 * batch[self.key] - 1
        return batch


class LightParamsMapper(BaseMapper):
    """
    Parse light parameters into a tensor-friendly list.

    Accepts JSON strings/bytes, dicts, lists, or tensors.
    """

    def __init__(self, config: LightParamsMapperConfig):
        super().__init__(config)
        self.expected_length = config.expected_length
        self.param_order = config.param_order

    def _parse_value(self, value: Any) -> List[float]:
        if isinstance(value, (bytes, bytearray)):
            value = value.decode("utf-8")
        if isinstance(value, str):
            value = json.loads(value)
        if isinstance(value, dict):
            return [float(value[key]) for key in self.param_order]
        if isinstance(value, (list, tuple)):
            return [float(v) for v in value]
        if hasattr(value, "tolist"):
            return [float(v) for v in value.tolist()]
        if isinstance(value, Iterable):
            return [float(v) for v in value]
        raise TypeError(f"Unsupported light params type: {type(value)}")

    def __call__(self, batch: Dict[str, Any], *args, **kwrags) -> Dict[str, Any]:
        if self.key in batch:
            values = self._parse_value(batch[self.key])
            if len(values) != self.expected_length:
                raise ValueError(
                    f"Expected {self.expected_length} light params, got {len(values)}"
                )
            batch[self.output_key] = values
        return batch
