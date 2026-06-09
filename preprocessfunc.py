"""
NetShield — Preprocessing Pipeline + Inference
===============================================
Converts raw NetShield API POST payloads into model-ready DataFrames
and runs the autoencoder to detect attacks.
"""

import os
import json
import pickle
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PROTO_MAP = {1: "ICMP", 6: "TCP", 17: "UDP"}

NUMERICAL_COLUMNS = [
    "frame length",
    "IP TTL",
    "IP Flags",
    "IP DSCP Field",
    "TCP SYN Flag",
    "TCP ACK Flag",
    "TCP FIN Flag",
    "TCP RST Flag",
    "TCP Window Size",
    "ICMP Type",
    "delta_time",
    "packet_ratio",
    "Destination Port",
    "has_encapsulation",
    "TCP PSH Flag",
    "is_tcp",
    "is_icmp",
    "is_http",
    "is_dns",
    "DNS_missing",
    "is_arp",
    "port_not_applicable",
    "port_present",
    "ttl_not_applicable",
    "ttl_true_missing",
    "ttl_present",
    "DSCP_malformed",
]

TEXTUAL_COLUMNS = [
    "Protocol",
    "IP Protocol",
    "HTTP Request Method",
    "HTTP Response Code",
    "HTTP Content-Length",
    "HTTP Content Type",
    "DNS Query Type",
]

MODEL_COLUMNS = NUMERICAL_COLUMNS + TEXTUAL_COLUMNS

# ---------------------------------------------------------------------------
# Global inference state (populated by load_model())
# ---------------------------------------------------------------------------

_autoencoder         = None
_encoder_for_targets = None
_scaler              = None
_lookups             = {}
_numerical_columns   = []
_textual_columns     = []
_threshold           = None


def load_model(save_dir: str = "netshield_model") -> None:
    global _autoencoder, _encoder_for_targets, _scaler
    global _lookups, _numerical_columns, _textual_columns, _threshold

    import tensorflow as tf
    from tensorflow.keras.layers import StringLookup

    print(f"Loading model artifacts from '{save_dir}/' ...")

    _autoencoder         = tf.keras.models.load_model(f"{save_dir}/autoencoder.keras")
    _encoder_for_targets = tf.keras.models.load_model(f"{save_dir}/encoder.keras")
    print("  ✓ Keras models loaded")

    with open(f"{save_dir}/scaler.pkl", "rb") as f:
        _scaler = pickle.load(f)
    print("  ✓ Scaler loaded")

    with open(f"{save_dir}/vocabularies.json") as f:
        vocabularies = json.load(f)

    _lookups = {
        col: StringLookup(
            vocabulary=vocab,
            output_mode="int",
            mask_token=None,
            num_oov_indices=1,
        )
        for col, vocab in vocabularies.items()
    }
    print(f"  ✓ {len(_lookups)} StringLookup layers rebuilt")

    with open(f"{save_dir}/meta.json") as f:
        meta = json.load(f)

    _numerical_columns = meta["numerical_columns"]
    _textual_columns   = meta["textual_columns"]
    _threshold         = meta["threshold"]
    print(f"  ✓ threshold={_threshold:.6f} | "
          f"num_cols={len(_numerical_columns)} cat_cols={len(_textual_columns)}")
    print("Model ready.\n")


def is_loaded() -> bool:
    return _autoencoder is not None


# ---------------------------------------------------------------------------
# Per-packet preprocessing
# ---------------------------------------------------------------------------

def _transform_packet(pkt: dict[str, Any], window_seconds: float, total_packets: int) -> dict:
    proto_num = pkt["ip_protocol"]
    proto_str = _PROTO_MAP.get(proto_num, str(proto_num))

    ip_ttl   = float(pkt["ip_ttl"])
    ip_dscp  = int(pkt["ip_dscp"])
    dst_port = float(pkt["dst_port"])

    tcp_syn = int(pkt["tcp_syn"])
    tcp_ack = int(pkt["tcp_ack"])
    tcp_fin = int(pkt["tcp_fin"])
    tcp_rst = int(pkt["tcp_rst"])
    tcp_psh = int(pkt["tcp_psh"])

    icmp_type = float(pkt["icmp_type"])

    http_method = pkt["http_method"]      or "NO_HTTP"
    http_code   = str(int(pkt["http_response_code"]))
    http_clen   = str(int(pkt["http_content_length"]) if pkt["http_content_length"] != 0 else -1)
    http_ctype  = pkt["http_content_type"] or "NO_HTTP"
    dns_qtype   = pkt["dns_query_type"]   or "NO_DNS"

    is_tcp  = int(proto_num == 6)
    is_icmp = int(proto_num == 1)
    is_http = int(bool(pkt["http_method"]) or pkt["http_response_code"] != 0)
    is_dns  = int(bool(pkt["dns_query_type"]))
    is_arp  = 0

    port_not_applicable = int(proto_num not in (6, 17))
    port_present        = int(not port_not_applicable)

    ttl_not_applicable = 0
    ttl_true_missing   = 0
    ttl_present        = 1

    dns_missing    = int(proto_num == 17 and dst_port == 53 and not bool(pkt["dns_query_type"]))
    dscp_malformed = int(ip_dscp != 0)

    ip_flags_combined = float((pkt["ip_flags_df"] << 1) | pkt["ip_flags_mf"])

    packet_ratio = total_packets / window_seconds if window_seconds > 0 else 0.0
    delta_time   = window_seconds / total_packets  if total_packets  > 0 else 0.0

    return {
        "frame length":        0,
        "IP TTL":              ip_ttl,
        "IP Flags":            ip_flags_combined,
        "IP DSCP Field":       ip_dscp,
        "TCP SYN Flag":        tcp_syn,
        "TCP ACK Flag":        tcp_ack,
        "TCP FIN Flag":        tcp_fin,
        "TCP RST Flag":        tcp_rst,
        "TCP Window Size":     -1.0,
        "ICMP Type":           icmp_type,
        "delta_time":          delta_time,
        "packet_ratio":        packet_ratio,
        "Destination Port":    dst_port,
        "has_encapsulation":   0,
        "TCP PSH Flag":        tcp_psh,
        "is_tcp":              is_tcp,
        "is_icmp":             is_icmp,
        "is_http":             is_http,
        "is_dns":              is_dns,
        "DNS_missing":         dns_missing,
        "is_arp":              is_arp,
        "port_not_applicable": port_not_applicable,
        "port_present":        port_present,
        "ttl_not_applicable":  ttl_not_applicable,
        "ttl_true_missing":    ttl_true_missing,
        "ttl_present":         ttl_present,
        "DSCP_malformed":      dscp_malformed,
        "Protocol":            proto_str,
        "IP Protocol":         proto_str,
        "HTTP Request Method": http_method,
        "HTTP Response Code":  http_code,
        "HTTP Content-Length": http_clen,
        "HTTP Content Type":   http_ctype,
        "DNS Query Type":      dns_qtype,
    }


def preprocess_window(payload: dict[str, Any]) -> pd.DataFrame:
    total_packets  = payload["total_packets"]
    window_seconds = payload["window_seconds"]
    rows = [
        _transform_packet(pkt, window_seconds, total_packets)
        for pkt in payload["features"]
    ]
    return pd.DataFrame(rows, columns=MODEL_COLUMNS)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict_window(df: pd.DataFrame) -> dict[str, Any]:
    if _autoencoder is None:
        raise RuntimeError("Call load_model() before predict_window().")

    num_data = _scaler.transform(df[_numerical_columns].values.astype("float32"))

    cat_data = [
        _lookups[col](df[col].astype(str).values).numpy()
        for col in _textual_columns
    ]

    inputs          = [num_data] + cat_data
    targets         = _encoder_for_targets.predict(inputs, verbose=0)
    reconstructions = _autoencoder.predict(inputs, verbose=0)

    mse = np.mean((reconstructions - targets) ** 2, axis=1)

    return {
        "is_attack":     bool(np.any(mse > _threshold)),
        "packet_scores": mse.tolist(),
        "threshold":     float(_threshold),
    }