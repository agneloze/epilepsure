from openenv.core import SyncEnvClient
from models import EpilepsyAction, EpilepsyObservation

class EpilepsyClient(SyncEnvClient):
    action_class = EpilepsyAction
    observation_class = EpilepsyObservation
