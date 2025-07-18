from abc import ABC, abstractmethod
from typing import Dict
from appPublic.log import debug, error, info, exception

connection_pathMap = {}

def connection_register(connection_key, Klass):
    """为给定的连接键注册一个连接类"""
    global connection_pathMap
    connection_pathMap[connection_key] = Klass
    info(f"Registered {connection_key} with class {Klass}")

def get_connection_class(connection_path):
    """根据连接路径查找对应的连接类"""
    global connection_pathMap
    debug(f"connection_pathMap: {connection_pathMap}")
    klass = connection_pathMap.get(connection_path)
    if klass is None:
        error(f"{connection_path} has not mapping to a connection class")
        raise Exception(f"{connection_path} has not mapping to a connection class")
    return klass

class BaseConnection(ABC):
    @abstractmethod
    async def handle_connection(self, action: str, params: Dict = None) -> Dict:
        """处理数据库操作，根据 action 执行创建集合等"""
        pass