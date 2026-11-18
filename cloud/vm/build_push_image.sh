#!/bin/bash
# Build the API image on the Azure VM and push it to Azure Container Registry.
# - ACR Tasks (`az acr build`) is not allowed on Azure for Students subscriptions, so we build on the VM.
# - Azure Container Apps "express" environments only accept OCI images, so we build with Podman
#   using --format oci (the legacy Docker builder produces Docker v2 manifests, which fail there).
# Login uses the VM's managed identity (role AcrPush on the registry): no password, no admin user.
#
# Usage on the VM, from the project root:
#   bash cloud/vm/build_push_image.sh acris402fl fraud-api v2
set -euo pipefail
REGISTRY=${1:-acris402fl}
IMAGE=${2:-fraud-api}
TAG=${3:-v1}
LOGIN_SERVER="$REGISTRY.azurecr.io"
REF="$LOGIN_SERVER/$IMAGE:$TAG"
cd "$(dirname "$0")/../../api"

if ! command -v podman >/dev/null; then
  sudo apt-get update -qq && sudo apt-get install -y -qq podman >/dev/null
fi
AUTHFILE=$(sudo mktemp)       # separate auth file owned by root (avoid any old docker login token)
echo '{}' | sudo tee "$AUTHFILE" >/dev/null
trap 'sudo rm -f "$AUTHFILE"' EXIT
PODMAN="sudo podman"
LOGIN_OPTS="--authfile $AUTHFILE"

# Managed identity -> Entra ID token -> ACR refresh token -> registry login
AAD_TOKEN=$(curl -s -H Metadata:true \
  "http://169.254.169.254/metadata/identity/oauth2/token?api-version=2018-02-01&resource=https://management.azure.com/" \
  | python3 -c "import sys, json; print(json.load(sys.stdin)['access_token'])")
ACR_TOKEN=$(curl -s -X POST "https://$LOGIN_SERVER/oauth2/exchange" \
  -d "grant_type=access_token&service=$LOGIN_SERVER&access_token=$AAD_TOKEN" \
  | python3 -c "import sys, json; print(json.load(sys.stdin)['refresh_token'])")
echo "$ACR_TOKEN" | $PODMAN login $LOGIN_OPTS "$LOGIN_SERVER" -u 00000000-0000-0000-0000-000000000000 --password-stdin >/dev/null
unset AAD_TOKEN ACR_TOKEN
echo "Đã đăng nhập $LOGIN_SERVER bằng managed identity"

T0=$(date +%s)
$PODMAN build -q --format oci -t "$REF" . >/dev/null
echo "Build (OCI) xong sau $(( $(date +%s) - T0 )) giây, kích thước: $($PODMAN image inspect "$REF" --format '{{.Size}}' | awk '{printf "%.0f MB", $1/1e6}')"

# Quick test of the container before pushing
$PODMAN rm -f api-test >/dev/null 2>&1 || true
$PODMAN run -d --name api-test -p 8000:8000 -e REGION=vm-test "$REF" >/dev/null
for i in $(seq 1 20); do curl -sf localhost:8000/health >/dev/null && break; sleep 1; done
echo "Kiểm tra container: $(curl -s localhost:8000/health)"
$PODMAN rm -f api-test >/dev/null

$PODMAN push -q $LOGIN_OPTS --format oci "$REF"
echo "Đã push $REF"
