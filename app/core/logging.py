import json
import logging

logger = logging.getLogger("model_router")


def log_request(data: dict):
    logger.info(json.dumps(data, ensure_ascii=False))
