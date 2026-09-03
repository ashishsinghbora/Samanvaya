# 🛡️ Defense-Grade Security Policy & Vulnerability Disclosure

As an aerospace-grade planetary photogrammetry and image registration framework engineered for ISRO's Chandrayaan missions, **Samanvaya (समन्वय)** adheres to rigorous Zero-Trust architectural principles and cryptographic integrity standards.

---

## 🔒 Supported Versions

Only active production and major release tracks receive defense-grade security updates, cryptographic patches, and dependency audits.

| Version | Status | Security Maintenance Level |
| :--- | :---: | :--- |
| **v2.x.x (Enterprise)** | :white_check_mark: **Supported** | Full active security patches, Merkle chain validation, and vulnerability triage. |
| **v1.x.x (Legacy)** | :x: **End of Life** | Deprecated. Users must upgrade to v2.0.0+. |
| **< v1.0.0** | :x: **Unsupported** | Prototype releases with no security guarantees. |

---

## 🛰️ Zero-Trust Security Architecture

Samanvaya incorporates several defense layers to protect planetary research pipelines against untrusted data ingestion and execution risks:

1. **Payload Ingestion Sandbox (`src/security/file_validator.py`)**:
   - Cryptographic magic-byte inspection (GeoTIFF, FITS, NASA PDS3/PDS4).
   - Hard memory allocation and dimension guards ($100{,}000 \times 100{,}000\text{ px}$, $20\text{ GB}$) preventing decompression and pixel-bomb exploits.
   - Defused XML parsing forbidding DTD expansion and external entity injection (XXE).
2. **Tamper-Evident Merkle Audit Trail (`src/security/audit.py`)**:
   - SHA-256 Merkle hash chains ensuring complete traceability of all input/output geospatial operations.
3. **Defense-in-Depth Containerization (`docker/Dockerfile.backend`)**:
   - Non-root user execution (`UID 1000`), read-only root filesystems, and strict seccomp profiles blocking unsafe syscalls (`ptrace`, `kexec_load`).
4. **Cryptographic Authentication & RBAC (`src/security/auth.py`)**:
   - RS256 asymmetric JWT authentication with role-based access control (Viewer, Operator, Administrator).

---

## 🚨 Reporting a Vulnerability

We deeply appreciate the efforts of security researchers, planetary scientists, and open-source contributors in keeping Samanvaya secure.

If you identify a security vulnerability or algorithmic exploit within this repository, **DO NOT file a public GitHub issue**.

### Responsible Disclosure Protocol:
1. **GitHub Private Vulnerability Reporting**:
   - Navigate to the **Security** tab of this repository.
   - Click on **Report a vulnerability** to open an encrypted private draft advisory.
2. **Direct Researcher Contact**:
   - Reach out privately to the maintainers with the prefix `[SECURITY-VULNERABILITY] Samanvaya`.

### What to Include in Your Advisory:
- Detailed description of the vulnerability, including impacted modules (e.g., Ingestion, Phase Congruency, FastAPI middleware).
- A reproducible Proof of Concept (PoC) or script.
- Potential impact assessment (e.g., Heap corruption, Denial of Service via pixel-bomb, XXE, unauthorized job submission).

---

## ⏱️ Response SLA & Triage Commitments

| Severity Rating (CVSS v3.1) | Initial Acknowledgment | Triage & Patch SLA |
| :--- | :---: | :---: |
| **Critical (9.0 - 10.0)** | $< 24\text{ hours}$ | $\le 48\text{ hours}$ |
| **High (7.0 - 8.9)** | $< 48\text{ hours}$ | $\le 7\text{ days}$ |
| **Medium (4.0 - 6.9)** | $< 72\text{ hours}$ | $\le 14\text{ days}$ |
| **Low (0.1 - 3.9)** | $< 5\text{ business days}$ | Next scheduled release |

All verified vulnerability reporters will be credited in our public security advisories and release notes upon successful patch deployment.
