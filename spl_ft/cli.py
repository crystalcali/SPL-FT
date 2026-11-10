import argparse
import json
import socket
from .config import home


def main():
    p = argparse.ArgumentParser(description="LFM3 experiment workbench")
    sub = p.add_subparsers(dest="command")
    web = sub.add_parser("web")
    web.add_argument("--host", default="0.0.0.0")
    web.add_argument("--port", type=int, default=7860)
    sub.add_parser("doctor")
    sub.add_parser("token")
    args = p.parse_args()
    if args.command == "doctor":
        from .tasks import doctor
        print(json.dumps(doctor(), indent=2))
    elif args.command == "token":
        from .app import access_token
        print(access_token(home()))
    else:
        import uvicorn
        from .app import create_app, access_token
        port = getattr(args, "port", 7860)
        print(f"\nSPL-FT: http://{socket.gethostname()}:{port}")
        print(f"Access key: {access_token(home())}\n")
        uvicorn.run(create_app(), host=getattr(args, "host", "0.0.0.0"), port=port)
