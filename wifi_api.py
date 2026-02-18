import time
import subprocess
import re
from flask import Blueprint, request, jsonify

bp = Blueprint("wifi_api", __name__)

WIFI_IF = "wlan0"
HOTSPOT_CON = "transmogrifier-hotspot"

def nmcli(args):
    # Run nmcli via sudo (your sudoers file already allows this)
    cmd = ["sudo", "/usr/bin/nmcli"] + args
    return subprocess.run(cmd, text=True, capture_output=True)

def valid_ip():
    r = subprocess.run(
        ["bash", "-lc", f"ip -4 addr show {WIFI_IF} | grep -qE 'inet (?!169\\.254\\.)'"],
        capture_output=True,
        text=True,
    )
    return r.returncode == 0

def con_name_for_ssid(ssid: str) -> str:
    # Connection names must be safe; SSIDs can contain spaces/special chars.
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", ssid).strip("_")[:40]
    return f"wifi_{safe}" if safe else "wifi_network"

@bp.get("/api/wifi/scan")
def scan():
    r = nmcli(["-t", "-f", "SSID,SIGNAL,SECURITY", "dev", "wifi", "list", "ifname", WIFI_IF])
    if r.returncode != 0:
        return jsonify({"ok": False, "error": (r.stderr or r.stdout).strip()}), 500

    nets = []
    for line in r.stdout.splitlines():
        if not line.strip():
            continue
        parts = line.split(":")
        ssid = parts[0] if len(parts) > 0 else ""
        signal = parts[1] if len(parts) > 1 else "0"
        security = parts[2] if len(parts) > 2 else ""
        if ssid:
            try:
                sig = int(signal)
            except ValueError:
                sig = 0
            nets.append({"ssid": ssid, "signal": sig, "security": security})

    nets.sort(key=lambda x: x["signal"], reverse=True)
    return jsonify({"ok": True, "networks": nets})

@bp.post("/api/wifi/connect")
def connect():
    data = request.get_json(force=True) or {}
    ssid = (data.get("ssid") or "").strip()
    password = data.get("password")  # can be None/"" for open networks

    if not ssid:
        return jsonify({"ok": False, "error": "SSID required"}), 400

    con_name = con_name_for_ssid(ssid)

    # Create the connection if it doesn't exist
    r = nmcli(["con", "show", con_name])
    if r.returncode != 0:
        r2 = nmcli(["con", "add", "type", "wifi", "ifname", WIFI_IF, "con-name", con_name, "ssid", ssid])
        if r2.returncode != 0:
            return jsonify({"ok": False, "error": (r2.stderr or r2.stdout).strip()}), 500

    # DHCP + autoconnect, moderate priority (your home profile can stay higher if desired)
    nmcli(["con", "mod", con_name, "ipv4.method", "auto"])
    nmcli(["con", "mod", con_name, "connection.autoconnect", "yes"])
    nmcli(["con", "mod", con_name, "connection.autoconnect-priority", "5"])

    # Configure security
    if password:
        r3 = nmcli(["con", "mod", con_name, "wifi-sec.key-mgmt", "wpa-psk"])
        if r3.returncode != 0:
            return jsonify({"ok": False, "error": (r3.stderr or r3.stdout).strip()}), 500
        r4 = nmcli(["con", "mod", con_name, "wifi-sec.psk", password])
        if r4.returncode != 0:
            return jsonify({"ok": False, "error": (r4.stderr or r4.stdout).strip()}), 500
    else:
        # open network
        nmcli(["con", "mod", con_name, "wifi-sec.key-mgmt", ""])
        nmcli(["con", "mod", con_name, "wifi-sec.psk", ""])

    # Attempt to switch off hotspot and join the new Wi-Fi
    nmcli(["-q", "con", "down", HOTSPOT_CON])

    r5 = nmcli(["con", "up", con_name])
    if r5.returncode != 0:
        # Revert to hotspot so the UI remains reachable
        nmcli(["-q", "con", "up", HOTSPOT_CON])
        return jsonify({"ok": False, "error": (r5.stderr or r5.stdout).strip()}), 400

    # Wait for DHCP lease
    for _ in range(12):
        if valid_ip():
            return jsonify({"ok": True, "message": "Connected", "ssid": ssid, "connection": con_name})
        time.sleep(1)

    # No IP => revert
    nmcli(["-q", "con", "up", HOTSPOT_CON])
    return jsonify({"ok": False, "error": "Connected but no valid IP. Reverted to hotspot."}), 400
