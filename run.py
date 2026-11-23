"""Launch TrafficLens with HTTPS so phone browsers allow camera access."""

from __future__ import annotations

import argparse
import logging
import sys
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Windows consoles often default to cp1252, which cannot encode the QR output or banner.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001 - cosmetic only
    pass

import uvicorn  # noqa: E402

from backend import config  # noqa: E402
from backend.certs import ensure_certificate, local_ipv4_addresses  # noqa: E402

logger = logging.getLogger("trafficlens")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the TrafficLens live car counter")
    parser.add_argument("--host", default=config.HOST, help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=None, help="Port (default: 8443 HTTPS / 8000 HTTP)")
    parser.add_argument(
        "--http",
        action="store_true",
        help="Serve plain HTTP. Camera access will only work on localhost, not on phones.",
    )
    parser.add_argument("--no-qr", action="store_true", help="Do not print a QR code")
    parser.add_argument("--open", action="store_true", dest="open_browser", help="Open the local URL")
    return parser.parse_args()


def print_qr(url: str) -> None:
    try:
        import qrcode

        qr = qrcode.QRCode(border=1)
        qr.add_data(url)
        qr.make(fit=True)
        qr.print_ascii(invert=True)
    except Exception as exc:  # noqa: BLE001 - QR output is optional
        logger.debug("Could not print QR code: %s", exc)


def rank_phone_ips(addresses: list[str]) -> list[str]:
    """Prefer a real LAN address over VirtualBox NAT / host-only addresses."""

    def score(ip: str) -> tuple[int, str]:
        if ip.startswith("10.0.2."):
            return (2, ip)  # VirtualBox NAT
        if ip.startswith("192.168.56."):
            return (3, ip)  # default VirtualBox host-only network
        return (0, ip)

    return sorted(addresses, key=score)


def main() -> int:
    args = parse_args()
    tls = not args.http
    port = args.port or (config.PORT if tls else 8000)
    scheme = "https" if tls else "http"

    cert_file = key_file = None
    if tls:
        try:
            cert_file, key_file = ensure_certificate()
        except Exception as exc:  # noqa: BLE001
            logger.error("Could not create the HTTPS certificate: %s", exc)
            logger.error("Retry with --http (localhost only) or check your Python packages.")
            return 1

    lan_ips = rank_phone_ips(local_ipv4_addresses())
    local_url = f"{scheme}://localhost:{port}"
    phone_urls = [f"{scheme}://{ip}:{port}" for ip in lan_ips]

    print("")
    print("  +----------------------------------------------------------+")
    print("  |  TrafficLens  -  live vehicle counter                    |")
    print("  +----------------------------------------------------------+")
    print("")
    print(f"  This computer : {local_url}")
    if phone_urls:
        print(f"  Your phone    : {phone_urls[0]}")
        for extra in phone_urls[1:]:
            print(f"                  {extra}  (alternative adapter)")
        print("                  (connect the phone to the same network)")
    else:
        print("  Your phone    : no LAN IPv4 address found")
    if tls:
        print("")
        print("  The certificate is self-signed. On the phone tap")
        print('  "Advanced" → "Proceed" once (Safari: "Show Details" → "visit this website").')
    else:
        print("")
        print("  HTTP mode: phone cameras are blocked by mobile browsers. Use localhost or enable HTTPS.")
    print("")
    print("  Press Ctrl+C to stop.")
    print("")

    if phone_urls and not args.no_qr:
        print("  Scan to open on your phone:")
        print_qr(phone_urls[0])
        print("")

    if args.open_browser:
        webbrowser.open(local_url)

    uvicorn.run(
        "backend.main:app",
        host=args.host,
        port=port,
        ssl_certfile=str(cert_file) if cert_file else None,
        ssl_keyfile=str(key_file) if key_file else None,
        log_level="info",
        access_log=False,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nStopped.")
        raise SystemExit(0)
    except OSError as exc:
        print(f"\nServer error: {exc}")
        raise SystemExit(1)
