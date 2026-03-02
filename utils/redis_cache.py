"""
Redis Cache Module
Handles caching of prediction results using Redis for faster repeated lookups.
"""

import json
import hashlib
import logging

logger = logging.getLogger(__name__)


class RedisCache:
    """
    Provides Redis-based caching for prediction results.
    Gracefully degrades if Redis is unavailable.
    
    Attributes:
        client: Redis client instance (or None if unavailable).
        ttl (int): Time-to-live for cache entries in seconds.
        is_available (bool): Whether Redis connection is active.
    """

    def __init__(self, host: str = 'localhost', port: int = 6379,
                 db: int = 0, password: str = None, ttl: int = 3600):
        """
        Constructor — attempts Redis connection with graceful fallback.
        
        Args:
            host (str): Redis server hostname.
            port (int): Redis server port.
            db (int): Redis database index.
            password (str): Redis password (optional).
            ttl (int): Cache entry time-to-live in seconds.
        """
        self.client = None
        self.ttl = ttl
        self.is_available = False

        try:
            import redis
            self.client = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=True,
                socket_timeout=3,
                socket_connect_timeout=3,
            )
            # Test connection
            self.client.ping()
            self.is_available = True
            logger.info("Redis connected at %s:%d (db=%d).", host, port, db)
        except ImportError:
            logger.warning("Redis package not installed. Caching disabled.")
        except Exception as e:
            logger.warning("Redis unavailable (%s). Caching disabled.", str(e))

    @staticmethod
    def _generate_key(image_bytes: bytes, model_key: str) -> str:
        """
        Generate a unique cache key from image content and model selection.
        
        Args:
            image_bytes (bytes): Raw bytes of the uploaded image.
            model_key (str): Model identifier string.
        
        Returns:
            str: SHA256-based cache key.
        """
        hasher = hashlib.sha256()
        hasher.update(image_bytes)
        hasher.update(model_key.encode('utf-8'))
        return f"food_pred:{hasher.hexdigest()}"

    def get(self, image_bytes: bytes, model_key: str) -> dict | None:
        """
        Retrieve cached prediction result.
        
        Args:
            image_bytes (bytes): Raw image bytes.
            model_key (str): Model key used for prediction.
        
        Returns:
            dict or None: Cached prediction result, or None if not found.
        """
        if not self.is_available:
            return None

        try:
            cache_key = self._generate_key(image_bytes, model_key)
            cached = self.client.get(cache_key)
            if cached:
                logger.info("Cache HIT for key: %s", cache_key[:20])
                return json.loads(cached)
            logger.info("Cache MISS for key: %s", cache_key[:20])
            return None
        except Exception as e:
            logger.warning("Cache read error: %s", str(e))
            return None

    def set(self, image_bytes: bytes, model_key: str, result: dict) -> bool:
        """
        Store prediction result in cache.
        
        Args:
            image_bytes (bytes): Raw image bytes.
            model_key (str): Model key used for prediction.
            result (dict): Prediction result to cache.
        
        Returns:
            bool: True if successfully cached, False otherwise.
        """
        if not self.is_available:
            return False

        try:
            cache_key = self._generate_key(image_bytes, model_key)
            self.client.setex(cache_key, self.ttl, json.dumps(result))
            logger.info("Cache SET for key: %s (TTL=%ds)", cache_key[:20], self.ttl)
            return True
        except Exception as e:
            logger.warning("Cache write error: %s", str(e))
            return False

    def flush(self) -> bool:
        """
        Clear all cached predictions.
        
        Returns:
            bool: True if flushed, False on error.
        """
        if not self.is_available:
            return False

        try:
            self.client.flushdb()
            logger.info("Redis cache flushed.")
            return True
        except Exception as e:
            logger.warning("Cache flush error: %s", str(e))
            return False

    def get_status(self) -> dict:
        """
        Get Redis connection status info.
        
        Returns:
            dict: Connection status details.
        """
        return {
            'connected': self.is_available,
            'ttl_seconds': self.ttl,
        }
