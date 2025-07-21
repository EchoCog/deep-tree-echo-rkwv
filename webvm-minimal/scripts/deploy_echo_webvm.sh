#!/bin/bash

# Deep Tree Echo WebVM Deployment Script
# Deploys the complete cognitive architecture with RWKV integration to WebVM

set -e

echo "=== Deep Tree Echo WebVM Deployment ==="
echo "Setting up cognitive architecture with RWKV integration..."

# Configuration
DEPLOYMENT_DIR="/home/ubuntu/echo_deployment"
WEBVM_DIR="$DEPLOYMENT_DIR/webvm_deployment"
ECHO_SOURCE="$DEPLOYMENT_DIR/echodeploy/echodash-main"
RWKV_LM_SOURCE="$DEPLOYMENT_DIR/RWKV-LM/RWKV-LM-main"
RWKV_RUNNER_SOURCE="$DEPLOYMENT_DIR/RWKV-Runner/RWKV-Runner-master"
RWKV_CPP_SOURCE="$DEPLOYMENT_DIR/rwkv-cpp/rwkv.cpp-master"
ARCHITECTURE_SOURCE="$DEPLOYMENT_DIR/architecture"

# Create WebVM project structure
echo "Creating WebVM project structure..."
mkdir -p "$WEBVM_DIR"/{src,assets,models,config,scripts}

# Copy WebVM base files
echo "Setting up WebVM base configuration..."
cp -r "$DEPLOYMENT_DIR/webvm/webvm-main"/* "$WEBVM_DIR/"

# Create custom index.html for Deep Tree Echo
cat > "$WEBVM_DIR/index.html" << 'EOF'
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Deep Tree Echo - Cognitive Architecture WebVM</title>
    <link rel="icon" type="image/x-icon" href="/favicon.ico">
    <style>
        body {
            margin: 0;
            padding: 0;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            color: white;
        }
        .header {
            text-align: center;
            padding: 20px;
            background: rgba(0,0,0,0.3);
        }
        .header h1 {
            margin: 0;
            font-size: 2.5em;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        }
        .header p {
            margin: 10px 0 0 0;
            font-size: 1.2em;
            opacity: 0.9;
        }
        .container {
            display: flex;
            height: calc(100vh - 120px);
        }
        .sidebar {
            width: 300px;
            background: rgba(0,0,0,0.2);
            padding: 20px;
            overflow-y: auto;
        }
        .main-content {
            flex: 1;
            padding: 20px;
        }
        .webvm-container {
            width: 100%;
            height: 100%;
            border: 2px solid rgba(255,255,255,0.3);
            border-radius: 8px;
            background: black;
        }
        .info-panel {
            background: rgba(255,255,255,0.1);
            padding: 15px;
            margin-bottom: 15px;
            border-radius: 8px;
            border-left: 4px solid #4CAF50;
        }
        .info-panel h3 {
            margin: 0 0 10px 0;
            color: #4CAF50;
        }
        .status-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: #4CAF50;
            margin-right: 8px;
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🧠 Deep Tree Echo</h1>
        <p>Membrane-Based Cognitive Architecture with RWKV Integration</p>
    </div>
    
    <div class="container">
        <div class="sidebar">
            <div class="info-panel">
                <h3><span class="status-indicator"></span>System Status</h3>
                <p>WebVM Environment: Active</p>
                <p>Cognitive Membranes: Initializing</p>
                <p>RWKV Models: Loading</p>
            </div>
            
            <div class="info-panel">
                <h3>🎪 Membrane Hierarchy</h3>
                <ul style="list-style: none; padding: 0;">
                    <li>🧠 Cognitive Membrane</li>
                    <li style="margin-left: 20px;">💭 Memory</li>
                    <li style="margin-left: 20px;">⚡ Reasoning</li>
                    <li style="margin-left: 20px;">🎭 Grammar</li>
                    <li>🔌 Extension Membrane</li>
                    <li style="margin-left: 20px;">🌍 Browser</li>
                    <li style="margin-left: 20px;">📊 ML</li>
                    <li style="margin-left: 20px;">🪞 Introspection</li>
                    <li>🛡️ Security Membrane</li>
                </ul>
            </div>
            
            <div class="info-panel">
                <h3>🤖 RWKV Integration</h3>
                <p>Model: RWKV-7 (1.5B parameters)</p>
                <p>Memory: Linear scaling</p>
                <p>Context: Infinite length</p>
                <p>Performance: Optimized for WebVM</p>
            </div>
            
            <div class="info-panel">
                <h3>🚀 Quick Start</h3>
                <p>1. Wait for system initialization</p>
                <p>2. Run: <code>./start_echo.sh</code></p>
                <p>3. Access dashboard at localhost:8000</p>
                <p>4. Begin cognitive interaction</p>
            </div>
        </div>
        
        <div class="main-content">
            <div id="webvm-container" class="webvm-container"></div>
        </div>
    </div>
    
    <script type="module">
        import { startWebVM } from './src/webvm.js';
        import config from './config/echo_webvm_config.js';
        
        // Initialize WebVM with Deep Tree Echo configuration
        startWebVM('webvm-container', config);
    </script>
</body>
</html>
EOF

# Create WebVM initialization module
cat > "$WEBVM_DIR/src/webvm.js" << 'EOF'
import { CheerpXApp } from '@leaningtech/cheerpx';

export async function startWebVM(containerId, config) {
    const container = document.getElementById(containerId);
    
    if (!container) {
        console.error('WebVM container not found');
        return;
    }
    
    try {
        console.log('Initializing Deep Tree Echo WebVM...');
        
        // Create CheerpX application with Echo configuration
        const app = await CheerpXApp.create({
            diskImage: config.diskImageUrl,
            diskImageType: config.diskImageType,
            cmd: config.cmd,
            args: config.args,
            env: config.opts.env,
            cwd: config.opts.cwd,
            uid: config.opts.uid,
            gid: config.opts.gid,
            enableNetworking: config.networkConfig?.enableTailscale || false,
            memorySize: config.opts.memorySize || 700 * 1024 * 1024
        });
        
        // Mount the application to the container
        app.mount(container);
        
        // Execute initialization script if provided
        if (config.initScript) {
            console.log('Running Deep Tree Echo initialization...');
            await app.run('/bin/bash', ['-c', config.initScript]);
        }
        
        console.log('Deep Tree Echo WebVM ready!');
        
        // Update status indicators
        updateSystemStatus('ready');
        
    } catch (error) {
        console.error('Failed to start Deep Tree Echo WebVM:', error);
        container.innerHTML = `
            <div style="color: red; padding: 20px; text-align: center;">
                <h3>Failed to initialize Deep Tree Echo WebVM</h3>
                <p>Error: ${error.message}</p>
                <p>Please check the console for more details.</p>
            </div>
        `;
    }
}

function updateSystemStatus(status) {
    const indicators = document.querySelectorAll('.status-indicator');
    indicators.forEach(indicator => {
        if (status === 'ready') {
            indicator.style.background = '#4CAF50';
        } else if (status === 'error') {
            indicator.style.background = '#f44336';
        }
    });
}
EOF

# Create package.json for the deployment
cat > "$WEBVM_DIR/package.json" << 'EOF'
{
    "name": "deep-tree-echo-webvm",
    "version": "1.0.0",
    "description": "Deep Tree Echo Cognitive Architecture deployed on WebVM with RWKV integration",
    "type": "module",
    "scripts": {
        "dev": "vite dev --host 0.0.0.0 --port 3000",
        "build": "vite build",
        "preview": "vite preview --host 0.0.0.0 --port 3000",
        "deploy": "npm run build && echo 'Deep Tree Echo WebVM build complete'"
    },
    "dependencies": {
        "@leaningtech/cheerpx": "latest",
        "@xterm/xterm": "^5.5.0",
        "@xterm/addon-fit": "^0.10.0",
        "@xterm/addon-web-links": "^0.11.0"
    },
    "devDependencies": {
        "vite": "^5.0.3",
        "vite-plugin-static-copy": "^1.0.6"
    },
    "keywords": ["cognitive-architecture", "webvm", "rwkv", "ai", "membrane-computing"],
    "author": "Deep Tree Echo Team",
    "license": "MIT"
}
EOF

# Create Vite configuration
cat > "$WEBVM_DIR/vite.config.js" << 'EOF'
import { defineConfig } from 'vite';
import { viteStaticCopy } from 'vite-plugin-static-copy';

export default defineConfig({
    plugins: [
        viteStaticCopy({
            targets: [
                {
                    src: 'assets/*',
                    dest: 'assets'
                },
                {
                    src: 'config/*',
                    dest: 'config'
                }
            ]
        })
    ],
    build: {
        outDir: 'dist',
        assetsDir: 'assets',
        rollupOptions: {
            input: {
                main: 'index.html'
            }
        }
    },
    server: {
        host: '0.0.0.0',
        port: 3000,
        cors: true
    },
    optimizeDeps: {
        include: ['@leaningtech/cheerpx']
    }
});
EOF

# Copy Echo system files
echo "Copying Deep Tree Echo system files..."
mkdir -p "$WEBVM_DIR/assets/echo_system"
cp -r "$ECHO_SOURCE"/* "$WEBVM_DIR/assets/echo_system/" 2>/dev/null || echo "Warning: Some Echo files may not exist"

# Copy RWKV implementations
echo "Copying RWKV implementations..."
mkdir -p "$WEBVM_DIR/assets/rwkv_models"
cp -r "$RWKV_LM_SOURCE"/* "$WEBVM_DIR/assets/rwkv_models/rwkv-lm/" 2>/dev/null || echo "Warning: RWKV-LM files may not exist"
cp -r "$RWKV_RUNNER_SOURCE"/* "$WEBVM_DIR/assets/rwkv_models/rwkv-runner/" 2>/dev/null || echo "Warning: RWKV-Runner files may not exist"
cp -r "$RWKV_CPP_SOURCE"/* "$WEBVM_DIR/assets/rwkv_models/rwkv-cpp/" 2>/dev/null || echo "Warning: RWKV.cpp files may not exist"

# Copy architecture documentation
echo "Copying architecture documentation..."
mkdir -p "$WEBVM_DIR/assets/docs"
cp -r "$ARCHITECTURE_SOURCE"/* "$WEBVM_DIR/assets/docs/" 2>/dev/null || echo "Warning: Architecture files may not exist"

# Create deployment README
cat > "$WEBVM_DIR/README.md" << 'EOF'
# Deep Tree Echo WebVM Deployment

This directory contains the complete WebVM deployment of the Deep Tree Echo cognitive architecture with RWKV integration.

## Quick Start

1. Install dependencies:
   ```bash
   npm install
   ```

2. Start development server:
   ```bash
   npm run dev
   ```

3. Build for production:
   ```bash
   npm run build
   ```

## Architecture

- **WebVM**: Browser-based Linux virtualization
- **Deep Tree Echo**: Membrane-based cognitive architecture
- **RWKV**: Efficient language model integration
- **Vite**: Modern build system

## Configuration

The system is configured through `config/echo_webvm_config.js` which includes:
- Memory optimization for WebVM constraints
- RWKV model configuration
- Cognitive membrane setup
- Performance tuning

## Deployment

The built system can be deployed to any static hosting service or served locally for development and testing.
EOF

echo "=== WebVM Deployment Setup Complete ==="
echo "Location: $WEBVM_DIR"
echo "Next steps:"
echo "1. cd $WEBVM_DIR"
echo "2. npm install"
echo "3. npm run dev"
echo "4. Open browser to http://localhost:3000"

