# TrafficLens in an Ubuntu VM.
#
# The VM gets three network paths:
#   - NAT + forwarded port 8443 -> accessible on this PC at https://localhost:8443
#   - host-only 192.168.56.20    -> accessible from this PC only
#   - bridged DHCP               -> gives the phone a reachable LAN address
#
# VirtualBox does not expose the host's Intel GPU to the guest, so inference runs
# on the VM's CPU through OpenVINO.

Vagrant.configure("2") do |config|
  config.vm.box = "ubuntu/jammy64"
  config.vm.hostname = "trafficlens-vm"
  config.vm.boot_timeout = 900

  # Local access from the Windows host (avoids the host firewall for localhost).
  config.vm.network "forwarded_port", guest: 8443, host: 8443, host_ip: "127.0.0.1", auto_correct: true

  # Stable address for host-only access.
  config.vm.network "private_network", ip: "192.168.56.20"

  # Phone access. Remove this line if corporate port security blocks the VM's MAC;
  # the app still works through the forwarded port on this PC.
  config.vm.network "public_network", bridge: "Intel(R) Ethernet Connection (17) I219-LM"

  config.vm.provider "virtualbox" do |vb|
    vb.name = "trafficlens-vm"
    vb.memory = "8192"
    vb.cpus = 12
    vb.gui = false
    vb.customize ["modifyvm", :id, "--ioapic", "on"]
    vb.customize ["modifyvm", :id, "--natdnshostresolver1", "on"]
  end

  # The Windows project folder is the single source of truth.
  config.vm.synced_folder ".", "/home/vagrant/car-counter", owner: "vagrant", group: "vagrant"

  config.vm.provision "shell", inline: <<~SHELL
    set -euo pipefail
    export DEBIAN_FRONTEND=noninteractive

    # The default Ubuntu mirrors are extremely slow on this network.
    sed -i 's|http://archive.ubuntu.com/ubuntu|https://mirrors.tuna.tsinghua.edu.cn/ubuntu|g; s|http://security.ubuntu.com/ubuntu|https://mirrors.tuna.tsinghua.edu.cn/ubuntu|g' /etc/apt/sources.list

    apt-get update -y
    apt-get install -y --no-install-recommends \
      python3 python3-venv libgl1 libglib2.0-0 libgomp1 libsm6 libxext6

    VENV=/home/vagrant/trafficlens-venv
    if [ ! -x "$VENV/bin/python" ]; then
      python3 -m venv "$VENV"
    fi

    "$VENV/bin/python" -m pip install --upgrade pip wheel

    # Install torch's own dependencies from the mirror, then the CPU-only wheels.
    # This prevents pip from pulling the multi-gigabyte CUDA build on Linux.
    "$VENV/bin/python" -m pip install \
      filelock typing-extensions sympy networkx jinja2 fsspec numpy pillow \
      --index-url https://pypi.tuna.tsinghua.edu.cn/simple

    "$VENV/bin/python" -m pip install --no-deps torch torchvision \
      --index-url https://download.pytorch.org/whl/cpu

    "$VENV/bin/python" -m pip install -r /home/vagrant/car-counter/requirements.txt \
      --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
      --extra-index-url https://pypi.org/simple

    "$VENV/bin/python" - <<'PY'
import cv2, openvino, torch, ultralytics
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("ultralytics", ultralytics.__version__, "openvino", openvino.__version__, "opencv", cv2.__version__)
PY

    cat > /etc/systemd/system/trafficlens.service <<'EOF'
[Unit]
Description=TrafficLens live vehicle counter
After=network-online.target local-fs.target
Wants=network-online.target
RequiresMountsFor=/home/vagrant/car-counter

[Service]
Type=simple
User=vagrant
WorkingDirectory=/home/vagrant/car-counter
Environment=CAR_COUNTER_DEVICE=intel:cpu
Environment=PYTHONUNBUFFERED=1
Environment=PYTHONDONTWRITEBYTECODE=1
ExecStartPre=/bin/sleep 8
ExecStart=/home/vagrant/trafficlens-venv/bin/python /home/vagrant/car-counter/run.py --no-qr
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

    systemctl daemon-reload
    systemctl enable trafficlens.service
    systemctl restart trafficlens.service
    sleep 8
    systemctl --no-pager --full status trafficlens.service || true
    echo "=== VM addresses ==="
    hostname -I || true
    echo "=== TrafficLens VM ready ==="
  SHELL
end
