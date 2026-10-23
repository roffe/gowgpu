// Copyright 2025 The GoGPU Authors
// SPDX-License-Identifier: MIT

package pipelinecache

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
)

const cacheRoot = "gowgpu"

// blobMagic starts every file SaveBlob writes, followed by the length of
// the blob and its SHA-256, so that LoadBlob can tell a whole blob from a
// torn or foreign file.
var blobMagic = [8]byte{'g', 'o', 'w', 'g', 'p', 'u', 'p', 'c'}

const blobHeaderSize = len(blobMagic) + 8 + sha256.Size

// UserCachePath returns an adapter-scoped cache file path under os.UserCacheDir().
// Example: ~/.cache/gowgpu/vulkan/<adapterKey>/pipeline.cache
func UserCachePath(backend, adapterKey, fileName string) (string, error) {
	dir, err := os.UserCacheDir()
	if err != nil {
		return "", fmt.Errorf("pipelinecache: UserCacheDir: %w", err)
	}
	return filepath.Join(dir, cacheRoot, backend, adapterKey, fileName), nil
}

// LoadBlob reads a cache blob SaveBlob wrote. A missing file returns
// (nil, nil); a file that is not a whole blob returns an error.
func LoadBlob(path string) ([]byte, error) {
	data, err := os.ReadFile(path) //nolint:gosec // path is constructed internally from UserCacheDir
	if os.IsNotExist(err) {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("pipelinecache: read %s: %w", path, err)
	}
	if len(data) < blobHeaderSize || [8]byte(data) != blobMagic ||
		binary.LittleEndian.Uint64(data[len(blobMagic):]) != uint64(len(data)-blobHeaderSize) ||
		[sha256.Size]byte(data[len(blobMagic)+8:]) != sha256.Sum256(data[blobHeaderSize:]) {
		return nil, fmt.Errorf("pipelinecache: %s is not a whole cache blob", path)
	}
	return data[blobHeaderSize:], nil
}

// SaveBlob writes a cache blob to disk through a temporary file of its own,
// synced before it is renamed over path, so that a reader sees the old blob
// or the new one whole, whoever else saves at the same time.
func SaveBlob(path string, data []byte) error {
	if len(data) == 0 {
		return nil
	}
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o750); err != nil {
		return fmt.Errorf("pipelinecache: mkdir %s: %w", dir, err)
	}
	f, err := os.CreateTemp(dir, filepath.Base(path)+".*.tmp")
	if err != nil {
		return fmt.Errorf("pipelinecache: create temp in %s: %w", dir, err)
	}
	var header [blobHeaderSize]byte
	copy(header[:], blobMagic[:])
	binary.LittleEndian.PutUint64(header[len(blobMagic):], uint64(len(data)))
	sum := sha256.Sum256(data)
	copy(header[len(blobMagic)+8:], sum[:])
	_, err = f.Write(header[:])
	if err == nil {
		_, err = f.Write(data)
	}
	if err == nil {
		err = f.Sync()
	}
	if cerr := f.Close(); err == nil {
		err = cerr
	}
	if err != nil {
		_ = os.Remove(f.Name())
		return fmt.Errorf("pipelinecache: write %s: %w", f.Name(), err)
	}
	if err := os.Rename(f.Name(), path); err != nil {
		_ = os.Remove(f.Name())
		return fmt.Errorf("pipelinecache: rename %s: %w", path, err)
	}
	return nil
}

// DeleteBlob removes a cache blob. Missing files are ignored.
func DeleteBlob(path string) error {
	err := os.Remove(path)
	if os.IsNotExist(err) {
		return nil
	}
	return err
}

// HexKey returns a hex-encoded SHA-256 digest of the given byte slices.
func HexKey(parts ...[]byte) string {
	h := sha256.New()
	for _, part := range parts {
		if len(part) > 0 {
			h.Write(part)
		}
	}
	return hex.EncodeToString(h.Sum(nil))
}

// VulkanAdapterKey identifies a Vulkan physical device for cache scoping.
func VulkanAdapterKey(vendorID, deviceID, driverVersion uint32, uuid [16]byte) string {
	var buf [28]byte
	binary.LittleEndian.PutUint32(buf[0:4], vendorID)
	binary.LittleEndian.PutUint32(buf[4:8], deviceID)
	binary.LittleEndian.PutUint32(buf[8:12], driverVersion)
	copy(buf[12:28], uuid[:])
	return HexKey(buf[:])
}

// DX12AdapterKey identifies a DXGI adapter for cache scoping.
func DX12AdapterKey(luidLow uint32, luidHigh int32, vendorID, deviceID, revision uint32) string {
	var buf [20]byte
	binary.LittleEndian.PutUint32(buf[0:4], luidLow)
	binary.LittleEndian.PutUint32(buf[4:8], uint32(luidHigh)) //nolint:gosec // LUID high half stored as raw bits
	binary.LittleEndian.PutUint32(buf[8:12], vendorID)
	binary.LittleEndian.PutUint32(buf[12:16], deviceID)
	binary.LittleEndian.PutUint32(buf[16:20], revision)
	return HexKey(buf[:])
}
