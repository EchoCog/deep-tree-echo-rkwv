# WebVM Integration for Deep Tree Echo

This directory contains the essential WebVM configuration files for deploying Deep Tree Echo in a browser-based Linux virtual machine.

## 🌐 What is WebVM?

WebVM enables running the Deep Tree Echo cognitive architecture directly in any web browser without requiring local installation. The current live deployment at https://lnh8imcjgdz8.manus.space is running on WebVM infrastructure.

## 📁 Essential Files

```
webvm-minimal/
├── config/
│   └── echo_webvm_config.js     # WebVM configuration for Deep Tree Echo
├── scripts/
│   └── deploy_echo_webvm.sh     # WebVM deployment script
└── README.md                    # This file
```

## 🚀 Quick WebVM Deployment

1. **Make deployment script executable**
```bash
chmod +x scripts/deploy_echo_webvm.sh
```

2. **Deploy to WebVM**
```bash
./scripts/deploy_echo_webvm.sh
```

3. **Access via browser**
The script will provide a WebVM URL where you can access Deep Tree Echo

## ⚙️ Configuration

The `config/echo_webvm_config.js` file contains WebVM-specific settings optimized for the 600MB memory limit and browser environment.

## 🎯 WebVM Benefits

✅ **Universal Access**: Works on any device with a modern browser  
✅ **Zero Installation**: No local setup required  
✅ **Sandboxed Security**: Isolated execution environment  
✅ **Cross-Platform**: Consistent experience across all platforms  

## 📚 Additional Resources

- [WebVM Official Documentation](https://webvm.io)
- [Deep Tree Echo Architecture](../docs/architecture/)
- [Live Demo](https://lnh8imcjgdz8.manus.space)

---

**WebVM makes Deep Tree Echo universally accessible! 🌐🧠**

