#!/usr/bin/env bash
set -euo pipefail

deploy_now_utc() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

deploy_now_utc_compact() {
  date -u +"%Y%m%dT%H%M%SZ"
}

deploy_log() {
  local level="$1"
  shift
  printf '[deploy][%s][%s] %s\n' "$(deploy_now_utc)" "${level}" "$*"
}

deploy_die() {
  deploy_log "ERROR" "$*"
  exit 1
}

deploy_parse_map_value() {
  local mapping="$1"
  local key="$2"
  local default_value="$3"
  local pair entry_key entry_value
  IFS=',' read -r -a pairs <<< "${mapping}"
  for pair in "${pairs[@]}"; do
    [[ -z "${pair}" ]] && continue
    entry_key="${pair%%=*}"
    entry_value="${pair#*=}"
    if [[ "${entry_key}" == "${key}" && "${entry_value}" != "${pair}" ]]; then
      printf '%s' "${entry_value}"
      return 0
    fi
  done
  printf '%s' "${default_value}"
}

deploy_worker_user() {
  local worker="$1"
  local default_user="${WORKER_SSH_USER:-ubuntu}"
  deploy_parse_map_value "${WORKER_SSH_USER_MAP:-}" "${worker}" "${default_user}"
}

deploy_worker_port() {
  local worker="$1"
  local default_port="${WORKER_SSH_PORT:-22}"
  deploy_parse_map_value "${WORKER_SSH_PORT_MAP:-}" "${worker}" "${default_port}"
}

deploy_worker_repo_root() {
  local worker="$1"
  local template="$2"
  local worker_user
  worker_user="$(deploy_worker_user "${worker}")"
  template="${template//\{WORKER\}/${worker}}"
  template="${template//\{WORKER_USER\}/${worker_user}}"
  printf '%s' "${template}"
}

deploy_lookup_worker_ip() {
  local topology_file="$1"
  local worker="$2"
  awk -F':' -v worker="${worker}" '
    $1 ~ "^"worker"$" {
      gsub(/^[ \t]+|[ \t]+$/, "", $2);
      print $2;
      found=1;
      exit 0;
    }
    END {
      if (!found) exit 1;
    }
  ' "${topology_file}"
}

deploy_encode_b64() {
  python3 -c 'import base64,sys; print(base64.b64encode(sys.argv[1].encode()).decode())' "$1"
}

deploy_write_kv_file() {
  local output_file="$1"
  shift
  mkdir -p "$(dirname "${output_file}")"
  : > "${output_file}"
  while [[ $# -gt 1 ]]; do
    local key="$1"
    local value="$2"
    shift 2
    printf '%s=%s\n' "${key}" "${value}" >> "${output_file}"
  done
}

deploy_replace_tokens() {
  local template="$1"
  shift
  while [[ $# -gt 1 ]]; do
    local key="$1"
    local value="$2"
    shift 2
    template="${template//\{$key\}/${value}}"
  done
  printf '%s' "${template}"
}

deploy_write_status() {
  local status_file="$1"
  local state="$2"
  local worker="$3"
  local experiment="$4"
  local mode="$5"
  local repeat="$6"
  local stage="$7"
  local detail="$8"
  mkdir -p "$(dirname "${status_file}")"
  cat > "${status_file}" <<EOF
state=${state}
worker=${worker}
experiment=${experiment}
mode=${mode}
repeat=${repeat}
stage=${stage}
detail=${detail}
updated_at=$(deploy_now_utc)
EOF
}

deploy_read_status_field() {
  local status_file="$1"
  local field="$2"
  awk -F'=' -v field="${field}" '$1 == field { print $2; exit 0 }' "${status_file}"
}

deploy_ssh_cmd() {
  local worker="$1"
  local host_ip="$2"
  local remote_cmd="$3"
  local user port
  user="$(deploy_worker_user "${worker}")"
  port="$(deploy_worker_port "${worker}")"

  local ssh_base=(ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p "${port}")

  if [[ -n "${WORKER_SSH_PASSWORD:-}" ]]; then
    sshpass -p "${WORKER_SSH_PASSWORD}" "${ssh_base[@]}" "${user}@${host_ip}" "${remote_cmd}"
  else
    "${ssh_base[@]}" "${user}@${host_ip}" "${remote_cmd}"
  fi
}
