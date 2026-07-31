import socket
import json

import numpy as np
from scipy.spatial.transform import Rotation as R


def call_remote(index, value, host="10.0.0.10", port=5005, timeout=5):
    with socket.create_connection((host, port), timeout=timeout) as sock:
        req = {"index": index, "value": value}
        print("Request sent:", req)
        sock.sendall(json.dumps(req).encode("utf-8"))
        data = sock.recv(4096).decode("utf-8")
        
        try:
            resp = json.loads(data)
            print("Received response:", resp)


            if isinstance(resp, dict):
                if resp.get("code") == 0:
                    return resp["result"]
                else:
                    raise RuntimeError(f"Server error: {resp.get('error')}")
            else:
                raise TypeError("Response is not a dictionary, it is a list or other type.")
        except Exception as e:
            print(f"Error decoding or processing response: {e}")
            raise



