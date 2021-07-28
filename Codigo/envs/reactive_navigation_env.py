# REACTIVE NAVIGATION - ENVIRONMENT
# Developed by: Luna Jimenez Fernandez
#
# This file contains the definition for an environment specific for the training
# of the Reactive Navigation Agent (using our proposed algorithm)
#
# Note that this environment is a variation of the already existing NavRLEnv
# (provided by the Habitat Baselines), that is adapted to use the proposed reward system
#
# The reward system works as follows:
# TODO: explica con mas detalle el algoritmo

# IMPORTS #
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.environments import NavRLEnv


@baseline_registry.register_env(name="ReactiveNavEnv")
class ReactiveNavigationEnv(NavRLEnv):
    """

    """
    # TODO: Definicion

    # ATTRIBUTES #
    # TODO: PON LOS ATRIBUTOS

    # CONSTRUCTOR #

    # PUBLIC METHODS #
    def get_reward_range(self):
        pass

    def get_reward(self, observations):
        pass
        # TODO: pon recompensa extra por exito? y penalizacion extra por fallo
        # TODO: ESTO VA AL CONFIG
