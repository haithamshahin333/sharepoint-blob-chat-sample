#!/bin/bash

echo "=== Azure Functions Core Tools Installation ==="

# Install Azure Functions Core Tools via APT
echo "Adding Microsoft GPG key..."
curl -sSL https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > microsoft.gpg
sudo mv microsoft.gpg /etc/apt/trusted.gpg.d/microsoft.gpg

echo "Adding Microsoft repository for Debian 11 (bullseye)..."
sudo sh -c 'echo "deb [arch=amd64] https://packages.microsoft.com/repos/microsoft-debian-bullseye-prod bullseye main" > /etc/apt/sources.list.d/dotnetdev.list'

echo "Updating package list..."
sudo apt-get update

echo "Installing Azure Functions Core Tools v4..."
sudo apt-get install -y azure-functions-core-tools-4

# Verify installation
echo "Verifying installation..."
if command -v func &> /dev/null; then
    echo "✅ Azure Functions Core Tools installed successfully!"
    echo "Version: $(func --version)"
else
    echo "❌ Installation failed. Trying alternative method..."
    
    # Fallback to npm installation
    echo "Installing via npm..."
    npm install -g azure-functions-core-tools@4 --unsafe-perm true
    
    if command -v func &> /dev/null; then
        echo "✅ Azure Functions Core Tools installed via npm!"
        echo "Version: $(func --version)"
    else
        echo "❌ Both installation methods failed. Please install manually."
        exit 1
    fi
fi

echo "=== Installation Complete ==="
