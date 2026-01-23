#!/bin/bash
# GCS Mount Setup Script for Scale-RAE Training
# This script mounts a Google Cloud Storage bucket to access WebDataset format training data

# Configuration
GCS_BUCKET="your-gcs-bucket-name"  # Replace with your GCS bucket name
MOUNT_POINT="/mnt/data"            # Local mount point

# Function to check if directory is mounted
is_mounted() {
    mount | grep -q "$MOUNT_POINT"
}

# Function to unmount if already mounted
cleanup_mount() {
    if is_mounted; then
        echo "Directory already mounted, unmounting..."
        sudo umount $MOUNT_POINT 2>/dev/null || true
        sleep 2
    fi
    
    # Kill any existing gcsfuse processes
    sudo pkill -f gcsfuse 2>/dev/null || true
    sleep 2
}

# Function to install gcsfuse
install_gcsfuse() {
    echo "🔧 Adding GCSFuse repository and key..."
    
    # Add Google Cloud packages key
    curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | \
        gpg --dearmor | \
        sudo tee /usr/share/keyrings/cloud.google.gpg > /dev/null

    # Add gcsfuse repository
    CODENAME=$(lsb_release -c -s)
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt gcsfuse-${CODENAME} main" | \
        sudo tee /etc/apt/sources.list.d/gcsfuse.list

    echo "🔄 Updating apt sources..."
    sudo apt-get update

    # Wait for dpkg lock to release
    echo "⏳ Waiting for dpkg/apt lock to be released..."
    while sudo fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1 || \
          sudo fuser /var/lib/apt/lists/lock >/dev/null 2>&1 || \
          sudo fuser /var/cache/apt/archives/lock >/dev/null 2>&1
    do
      echo "🔒 Lock is held by another process. Sleeping for 5s..."
      sleep 5
    done
    echo "✅ Lock is free. Continuing..."

    echo "🛠️ Configuring dpkg..."
    sudo dpkg --configure -a || true

    echo "📦 Installing fuse and gcsfuse..."
    sudo apt-get install -y fuse gcsfuse

    # Enable user_allow_other in fuse.conf
    if ! grep -q "^user_allow_other" /etc/fuse.conf 2>/dev/null; then
        echo "🔧 Enabling user_allow_other in /etc/fuse.conf"
        echo "user_allow_other" | sudo tee -a /etc/fuse.conf
    fi
}

# Function to setup mount directory
setup_directory() {
    echo "Setting up mount directory..."
    if [ ! -d "$MOUNT_POINT" ]; then
        sudo mkdir -p $MOUNT_POINT
    fi
    
    sudo chmod -R 777 $MOUNT_POINT
    
    if [ ! -d "$MOUNT_POINT" ]; then
        echo "Error: Failed to create mount directory"
        exit 1
    fi
}

# Function to create gcsfuse config
create_config() {
    echo "Creating gcsfuse configuration..."
    cat > $HOME/gcsfuse_config << 'EOF'
file-cache:
  max-size-mb: 20480  # 20GB cache
  cache-file-for-range-read: false

metadata-cache:
  stat-cache-max-size-mb: 32
  ttl-secs: 20
  type-cache-max-size-mb: 4

cache-dir: /tmp/gcsfuse
EOF
}

# Function to mount with retry logic
mount_with_retry() {
    echo "Attempting to mount gcsfuse..."
    local max_retries=3
    local retry_count=0
    
    while [ $retry_count -lt $max_retries ]; do
        echo "Mount attempt $((retry_count + 1))/$max_retries"
        
        cleanup_mount
        sleep 2
        
        # Attempt to mount
        if gcsfuse -o nonempty --implicit-dirs --config-file $HOME/gcsfuse_config $GCS_BUCKET $MOUNT_POINT; then
            sleep 5
            
            # Verify mount was successful
            if is_mounted && [ -d "$MOUNT_POINT" ] && timeout 15 ls $MOUNT_POINT > /dev/null 2>&1; then
                echo "✅ Mount successful!"
                echo "Mount point: $MOUNT_POINT"
                echo "Contents check passed"
                return 0
            else
                echo "❌ Mount verification failed"
            fi
        else
            echo "❌ Mount command failed"
        fi
        
        retry_count=$((retry_count + 1))
        if [ $retry_count -lt $max_retries ]; then
            echo "Retrying in 10 seconds..."
            sleep 10
        fi
    done
    
    echo "❌ Failed to mount after $max_retries attempts"
    return 1
}

# Main execution
echo "🚀 Starting GCS mount setup..."

# Check if gcsfuse is installed
if ! command -v gcsfuse &> /dev/null; then
    echo "gcsfuse not found, installing..."
    install_gcsfuse
else
    echo "gcsfuse already installed"
fi

# Early exit if already mounted
if is_mounted; then
    echo "✅ $MOUNT_POINT is already mounted. Nothing to do."
    exit 0
fi

cleanup_mount
setup_directory
create_config

# Mount with retry
if mount_with_retry; then
    echo "🎉 Setup completed successfully!"
    echo "You can now access your GCS bucket at: $MOUNT_POINT"
else
    echo "💥 Setup failed. Please check your GCS permissions and try again."
    exit 1
fi
