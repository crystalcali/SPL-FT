"""Isolated, non-executing UI fixture for development screenshots only.

Not used by start.sh, installed CLI or production. No real credentials are
provided to the browser. Production authentication is tested via TestClient.
"""
import os
from pathlib import Path
import uvicorn
from spl_ft.app import create_app, access_token
root = Path(os.environ['SPL_FT_HOME'])
app = create_app(root, start_worker=False)
key = access_token(root)

@app.middleware('http')
async def development_fixture(request, call_next):
    request.scope['headers'] = [(k, v) for k, v in request.scope['headers'] if k != b'cookie'] + [(b'cookie', ('spl_session=' + key).encode())]
    return await call_next(request)

uvicorn.run(app, host='0.0.0.0', port=int(os.environ.get('PORT', '4173')))
