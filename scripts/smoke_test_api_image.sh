#!/usr/bin/env bash
# scripts/smoke_test_api_image.sh IMAGE
# Author: Monzia Moodie
#
# The API image's startup contract, in ONE place. Called by ci.yml's docker-build job (pull requests, main)
# and by push-ghcr (published releases), so the image that is PUBLISHED is the exact image that PASSED.
#
# DOCKERCOPY-1 (2026-08-07). The original smoke step checked for one field NAME in the /health response,
# which passed on ANY response containing it; a service that started and lied went green. The image ships
# NO model artifact (/models/ is gitignored), so the honest contract is ALIVE AND NOT READY: live true,
# ready false, model_loaded false, status degraded.
#
# /health is POLLED until it answers or SMOKE_TIMEOUT_SECONDS (default 60) pass -- a fixed sleep either
# wasted time or failed on a slow runner. The container is removed on every exit path.
set -euo pipefail

image="${1:?usage: smoke_test_api_image.sh IMAGE}"
timeout_seconds="${SMOKE_TIMEOUT_SECONDS:-60}"
name="gvc-smoke-api-$$"
health="$(mktemp)"

cleanup() {
  docker rm -f "$name" >/dev/null 2>&1 || true
  rm -f "$health"
}
trap cleanup EXIT

docker run --detach --name "$name" -p 8000:8000 "$image" >/dev/null

deadline=$(( SECONDS + timeout_seconds ))
until curl --fail --silent --show-error http://localhost:8000/health -o "$health"; do
  if (( SECONDS >= deadline )); then
    echo "::error::/health did not respond within ${timeout_seconds}s. Container logs:"
    docker logs "$name" || true
    exit 1
  fi
  sleep 2
done
cat "$health"
echo

if ! jq -e '.live == true
            and .ready == false
            and .model_loaded == false
            and .status == "degraded"' "$health" > /dev/null; then
  echo "::error::the model-less container did not report the expected health contract"
  docker logs "$name" || true
  exit 1
fi
echo "startup contract held: live, not ready, no model loaded, degraded"
