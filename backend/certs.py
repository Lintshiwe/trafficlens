"""Self-signed certificate generation for phone camera access over HTTPS."""

from __future__ import annotations

import ipaddress
import logging
import socket
from datetime import datetime, timedelta, timezone
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

from . import config

logger = logging.getLogger(__name__)


def local_ipv4_addresses() -> list[str]:
    """Best-effort list of this computer's LAN IPv4 addresses."""
    addresses: set[str] = set()

    # psutil sees every interface, which matters inside a VM with NAT + host-only + bridged NICs.
    try:
        import psutil

        for interface_addrs in psutil.net_if_addrs().values():
            for addr in interface_addrs:
                if addr.family == socket.AF_INET:
                    ip = str(addr.address)
                    if ip and not ip.startswith(("127.", "169.254.")):
                        addresses.add(ip)
    except Exception:  # noqa: BLE001 - psutil is optional at runtime
        pass

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(0.3)
        sock.connect(("8.8.8.8", 80))
        addresses.add(sock.getsockname()[0])
        sock.close()
    except OSError:
        pass

    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if not ip.startswith("127."):
                addresses.add(ip)
    except OSError:
        pass

    return sorted(a for a in addresses if not a.startswith("169.254."))


def _certificate_is_usable(cert_path: Path, required_ips: list[str]) -> bool:
    try:
        cert = x509.load_pem_x509_certificate(cert_path.read_bytes())
        not_after = getattr(cert, "not_valid_after_utc", None)
        if not_after is None:
            not_after = cert.not_valid_after.replace(tzinfo=timezone.utc)
        if not_after < datetime.now(timezone.utc) + timedelta(days=7):
            return False

        san = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
        covered = {str(ip) for ip in san.get_values_for_type(x509.IPAddress)}
        return all(ip in covered for ip in required_ips)
    except Exception:  # noqa: BLE001 - regenerate on any parsing problem
        return False


def ensure_certificate(force: bool = False) -> tuple[Path, Path]:
    """Create (or reuse) a self-signed cert covering localhost and LAN IPs."""
    config.CERT_DIR.mkdir(parents=True, exist_ok=True)
    cert_path = config.CERT_DIR / "cert.pem"
    key_path = config.CERT_DIR / "key.pem"
    addresses = local_ipv4_addresses()

    if not force and cert_path.exists() and key_path.exists() and _certificate_is_usable(cert_path, addresses):
        return cert_path, key_path

    logger.info("Generating a self-signed HTTPS certificate ...")
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name(
        [
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "TrafficLens"),
            x509.NameAttribute(NameOID.COMMON_NAME, "TrafficLens Local"),
        ]
    )

    san_entries: list[x509.GeneralName] = [
        x509.DNSName("localhost"),
        x509.DNSName(socket.gethostname()),
        x509.DNSName("trafficlens.local"),
    ]
    san_entries.append(x509.IPAddress(ipaddress.ip_address("127.0.0.1")))
    for address in addresses:
        try:
            san_entries.append(x509.IPAddress(ipaddress.ip_address(address)))
        except ValueError:
            continue

    now = datetime.now(timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=5))
        .not_valid_after(now + timedelta(days=825))
        .add_extension(x509.SubjectAlternativeName(san_entries), critical=False)
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=True,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=True,
                crl_sign=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
        .sign(key, hashes.SHA256())
    )

    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    logger.info("Certificate written to %s", cert_path)
    return cert_path, key_path
