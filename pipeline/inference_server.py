import dataclasses
import logging
import socket

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config

class PolicyMetadata:
    def __init__(self, config: str, checkpoint: str):
        self.config = config
        self.checkpoint = checkpoint

POLICY_DICT = {
    "ranger_fast": PolicyMetadata(
        config="pi0_fast_ranger",
        checkpoint="/home/airobot/zzh/openpi/checkpoints/pi0_fast_ranger/0621/7999"
    ),
}

@dataclasses.dataclass
class Args:
    env: str = "ranger_fast"
    port: int = 8000
    record: bool = False


def create_policy(args: Args) -> _policy.Policy:
    return _policy_config.create_trained_policy(
        _config.get_config(POLICY_DICT[args.env].config), 
        POLICY_DICT[args.env].checkpoint,
    )


def main() -> None:
    args = Args()
    policy = create_policy(args)
    policy_metadata = policy.metadata

    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
