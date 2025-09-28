
Run echo "Processing requests..." && \
Processing requests...
Launcher version: 21ff7f3f8b6af653b8f5506ece65f4fb5d26f0ca, 0.0.37, 1.4.4
==============================
Allow list
----
localhost
https://github.com/
githubusercontent.com
https://raw.githubusercontent.com/
https://objects.githubusercontent.com/
https://codeload.github.com/
https://uploads.github.com/user-attachments/assets/
https://api.github.com/internal/user-attachments/assets/
https://api.github.com/advisories
https://github.githubassets.com/assets
https://*.githubusercontent.com
https://uploads.github.com
172.18.0.1
168.63.129.16
host.docker.internal
https://lfs.github.com/
https://github-cloud.githubusercontent.com/
https://github-cloud.s3.amazonaws.com/
https://api.githubcopilot.com/
https://api.githubcopilot.com/

https://github.com
runnervmf4ws1
----
==============================
Allow list
----
localhost
https://github.com/
githubusercontent.com
https://raw.githubusercontent.com/
https://objects.githubusercontent.com/
https://codeload.github.com/
https://uploads.github.com/user-attachments/assets/
https://api.github.com/internal/user-attachments/assets/
https://api.github.com/advisories
https://github.githubassets.com/assets
https://*.githubusercontent.com
https://uploads.github.com
172.18.0.1
168.63.129.16
host.docker.internal
https://lfs.github.com/
https://github-cloud.githubusercontent.com/
https://github-cloud.s3.amazonaws.com/
https://api.githubcopilot.com/
https://api.githubcopilot.com/

https://github.com
runnervmf4ws1
----
==============================
Using Firewall Recommended Rules
----

---
version: 0.0.1
rules:
  - kind: ip-rule
    name: azure-metadata-ip
    ip: 168.63.129.16

---
version: 0.0.1
rules:
### Key Features
- **Multi-Backend Support**: Seamlessly switches between RWKV.cpp, pip package, and mock backends
- **Intelligent Fallback**: Automatically degrades gracefully when preferred backend isn't available
- **Unified API**: Same interface works regardless of underlying backend
- **Performance Monitoring**: Tracks backend status and performance metrics
- **Memory Operations**: Backend-specific optimizations for encoding/retrieval

### Resolved Conflicts
- **README.md**: Merged documentation to showcase both Simple RWKV (v1.2) and RWKV.cpp features
- **echo_rwkv_bridge.py**: Enhanced with multi-backend architecture and implemented missing abstract methods
- **All Python files**: Resolved syntax conflicts while preserving functionality from both branches

### Benefits
- 🚀 **Performance**: RWKV.cpp backend provides 10x speed improvement when available
- 🎯 **Simplicity**: `pip install rwkv` approach for quick setup and development
- 🔄 **Reliability**: Automatic fallback ensures system always works
- 📦 **Compatibility**: Maintains backward compatibility with existing code
- 🧪 **Testing**: Enhanced mock backend for development without model dependencies

### Validation
All changes have been thoroughly tested:
- ✅ Python syntax validation for all resolved files
- ✅ Import testing for all modules
- ✅ Multi-backend functionality verification
- ✅ Memory operations testing
- ✅ Response generation validation
- ✅ Documentation conflict resolution

This solution provides users with the flexibility to choose their preferred RWKV integration method while ensuring the system remains functional regardless of available dependencies.

Fixes #48.

forceExit is shutting down the process
