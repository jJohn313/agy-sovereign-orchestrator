#!/usr/bin/env bash
# ==============================================================================
# AGY Sovereign System-1 Orchestrator Deployment Script
# Deploys the full sovereign orchestrator architecture to the host machine.
# ==============================================================================

set -euo pipefail

DEST_DIR="${HOME}/.config/agy/orchestrator"
CONFIG_DIR="${HOME}/.config/agy"
BIN_DIR="${HOME}/.local/bin"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================================="
echo " Deploying AGY Sovereign System-1 Orchestrator"
echo "=========================================================="

# 1. Create directories
echo "[1/6] Creating system directories..."
mkdir -p "${DEST_DIR}" "${CONFIG_DIR}" "${BIN_DIR}" "${DEST_DIR}/tests"

# 2. Copy implementation modules
echo "[2/6] Copying core engine modules to ${DEST_DIR}..."
cp "${SCRIPT_DIR}/"*.py "${DEST_DIR}/"
cp "${SCRIPT_DIR}/requirements.txt" "${DEST_DIR}/"
cp "${SCRIPT_DIR}/tests/"*.py "${DEST_DIR}/tests/"

# Copy baseline configs if not present
if [ ! -f "${CONFIG_DIR}/mcps.json" ] && [ -f "${SCRIPT_DIR}/config/mcps.json" ]; then
    echo "       Installing baseline mcps.json..."
    cp "${SCRIPT_DIR}/config/mcps.json" "${CONFIG_DIR}/mcps.json"
fi

if [ ! -f "${CONFIG_DIR}/env" ] && [ -f "${SCRIPT_DIR}/config/env.example" ]; then
    echo "       Creating ~/.config/agy/env from template..."
    cp "${SCRIPT_DIR}/config/env.example" "${CONFIG_DIR}/env"
fi

# 3. Setup Virtual Environment
echo "[3/6] Setting up isolated Python runtime..."
VENV_PATH="${DEST_DIR}/.venv"

# Use uv if available, otherwise python3 -m venv
if command -v uv &>/dev/null; then
    echo "       Using 'uv' to manage virtual environment..."
    uv venv "${VENV_PATH}" --python 3.12 2>/dev/null || uv venv "${VENV_PATH}"
    uv pip install -r "${DEST_DIR}/requirements.txt" --python "${VENV_PATH}/bin/python3"
else
    echo "       Using standard 'python3 -m venv'..."
    python3 -m venv "${VENV_PATH}"
    "${VENV_PATH}/bin/pip" install --upgrade pip
    "${VENV_PATH}/bin/pip" install -r "${DEST_DIR}/requirements.txt"
fi

# Add orchestrator to Python site-packages path
SITE_PACKAGES=$(find "${VENV_PATH}/lib" -maxdepth 2 -type d -name "site-packages" | head -n 1)
if [ -n "${SITE_PACKAGES}" ]; then
    echo "${DEST_DIR}" > "${SITE_PACKAGES}/orchestrator.pth"
fi

# 4. Install Global Launcher Shim
echo "[4/6] Installing global launcher shim at ${BIN_DIR}/agy..."

# Safely handle existing agy binary
EXISTING_AGY=$(which agy 2>/dev/null || true)
if [ -n "${EXISTING_AGY}" ] && [ "${EXISTING_AGY}" != "${BIN_DIR}/agy" ]; then
    echo "       Found existing agy binary at ${EXISTING_AGY}"
    # If the existing agy is already a shim pointing to this orchestrator, don't rename it
    if ! grep -q "orchestrator.py" "${EXISTING_AGY}"; then
        EXISTING_AGY_DIR=$(dirname "${EXISTING_AGY}")
        AGY_BIN_PATH="${EXISTING_AGY_DIR}/agy-bin"
        echo "       Renaming existing agy to ${AGY_BIN_PATH}..."
        # Use sudo if it's in a system path
        if [ ! -w "${EXISTING_AGY_DIR}" ]; then
            sudo mv "${EXISTING_AGY}" "${AGY_BIN_PATH}"
        else
            mv "${EXISTING_AGY}" "${AGY_BIN_PATH}"
        fi
        export DETECTED_AGY_ORIGINAL_BIN="${AGY_BIN_PATH}"
    else
        echo "       Existing agy is already an orchestrator shim."
        export DETECTED_AGY_ORIGINAL_BIN="${AGY_ORIGINAL_BIN:-$(which agy-bin 2>/dev/null || echo '')}"
    fi
else
    # Check if agy-bin exists
    export DETECTED_AGY_ORIGINAL_BIN="${AGY_ORIGINAL_BIN:-$(which agy-bin 2>/dev/null || echo '')}"
fi

cat << EOF > "${BIN_DIR}/agy"
#!/usr/bin/env bash
# Global Jev System-1 Shim for agy
export AGY_ORIGINAL_BIN="\${AGY_ORIGINAL_BIN:-${DETECTED_AGY_ORIGINAL_BIN}}"
exec "${HOME}/.config/agy/orchestrator/.venv/bin/python3" "${HOME}/.config/agy/orchestrator/orchestrator.py" "\$@"
EOF
chmod +x "${BIN_DIR}/agy"

# 5. Ensure PATH includes ~/.local/bin
echo "[5/6] Ensuring ~/.local/bin is in PATH..."
for RC_FILE in "${HOME}/.bashrc" "${HOME}/.zshrc"; do
    if [ -f "${RC_FILE}" ]; then
        if ! grep -q 'PATH=.*\.local/bin' "${RC_FILE}"; then
            echo 'export PATH="$HOME/.local/bin:$PATH"' >> "${RC_FILE}"
            echo "       Added PATH export to ${RC_FILE}"
        fi
    fi
done

# 6. Self-Verification
echo "[6/6] Running sovereign verification suite..."
if "${VENV_PATH}/bin/python3" -m unittest discover -s "${DEST_DIR}/tests"; then
    echo "=========================================================="
    echo " Deployment Succeeded! All verification tests passed."
    echo " You can now run 'agy <prompt>' from any directory."
    echo "=========================================================="
else
    echo "Warning: Some verification tests encountered issues. Review logs above."
fi
