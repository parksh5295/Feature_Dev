"""NSL-KDD column names, optional Train+ difficulty column, and behavior groups."""

from __future__ import annotations

from typing import Dict, List

NSL_KDD_COLUMNS: List[str] = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    "label",
]

# Present only in KDDTrain+ when the file has one extra column after label.
NSL_OPTIONAL_DIFFICULTY = "difficulty"

# Map behavior names to numeric feature names (categorical columns omitted).
NSL_KDD_BEHAVIOR_GROUPS: Dict[str, List[str]] = {
    "Connection intensity": [
        "count",
        "srv_count",
        "dst_host_count",
        "dst_host_srv_count",
        "srv_diff_host_rate",
        "dst_host_srv_diff_host_rate",
    ],
    "Data volume": [
        "src_bytes",
        "dst_bytes",
        "wrong_fragment",
        "urgent",
    ],
    "Timing pattern": [
        "duration",
    ],
    "Protocol / error rates": [
        "serror_rate",
        "srv_serror_rate",
        "rerror_rate",
        "srv_rerror_rate",
        "same_srv_rate",
        "diff_srv_rate",
        "dst_host_same_srv_rate",
        "dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate",
        "dst_host_serror_rate",
        "dst_host_srv_serror_rate",
        "dst_host_rerror_rate",
        "dst_host_srv_rerror_rate",
    ],
    "Host access / shell": [
        "land",
        "hot",
        "num_failed_logins",
        "logged_in",
        "num_compromised",
        "root_shell",
        "su_attempted",
        "num_root",
        "num_file_creations",
        "num_shells",
        "num_access_files",
        "num_outbound_cmds",
        "is_host_login",
        "is_guest_login",
    ],
}

NSL_NORMAL_LABELS = frozenset({"normal"})
