from typing import Dict
from appPublic.log import debug, error, info

connection_pathMap = {}

def connection_register(connection_key, Klass):
    """为给定的数据库注册一个数据库类"""
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

class BaseDBConnection:
    async def handle_connection(self, action: str, params: Dict = None) -> Dict:
        """默认的数据库操作处理方法，子类可重写"""
        if params is None:
            params = {}
        return {
            "status": "error",
            "message": f"Action {action} not implemented in {self.__class__.__name__}",
            "collection_name": "",
            "document_id": "",
            "status_code": 400
        }