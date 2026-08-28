#!/usr/bin/env python

from dataclasses import dataclass, field

from lerobot.configs import FeatureType, PreTrainedConfig

from .configuration_act import ACTConfig


@PreTrainedConfig.register_subclass("act_force")
@dataclass
class ACTForceConfig(ACTConfig):
    """ACT variant that consumes force/torque observations as separate conditioning tokens.

    Force inputs are regular LeRobot numeric observation features. By default this policy expects a
    feature named ``observation.force``. You can override ``force_feature_keys`` to use another key or
    multiple keys, for example ``["observation.left_force", "observation.right_force"]``.
    """

    force_feature_keys: list[str] = field(default_factory=lambda: ["observation.force"])

    def validate_features(self) -> None:
        super().validate_features()

        if not self.input_features:
            raise ValueError("ACTForce requires input features.")
        if not self.force_feature_keys:
            raise ValueError("ACTForce requires at least one force feature key.")

        missing_keys = [key for key in self.force_feature_keys if key not in self.input_features]
        if missing_keys:
            raise ValueError(
                "ACTForce force features are missing from the dataset/config: "
                f"{missing_keys}. Add them as numeric observation features, e.g. 'observation.force'."
            )

        non_state_keys = [
            key for key in self.force_feature_keys if self.input_features[key].type is not FeatureType.STATE
        ]
        if non_state_keys:
            raise ValueError(
                "ACTForce force features must be numeric observation features with FeatureType.STATE. "
                f"Got non-state keys: {non_state_keys}."
            )
