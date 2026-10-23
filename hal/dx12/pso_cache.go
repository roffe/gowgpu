// Copyright 2025 The GoGPU Authors
// SPDX-License-Identifier: MIT

//go:build windows && !(js && wasm)

package dx12

import (
	"encoding/binary"
	"errors"
	"path/filepath"
	"runtime"
	"unsafe"

	"github.com/roffe/gowgpu/hal/dx12/d3d12"
	"github.com/roffe/gowgpu/hal/dx12/dxgi"
	"github.com/roffe/gowgpu/internal/pipelinecache"
)

// PSOBlobStore persists driver-compiled PSO blobs keyed by descriptor hash.
type PSOBlobStore struct {
	dir string
}

// createGraphicsPSO creates a graphics PSO, loading/saving cached driver blobs.
func (d *Device) createGraphicsPSO(
	psoDesc *d3d12.D3D12_GRAPHICS_PIPELINE_STATE_DESC,
	cacheKey string,
	cachedBlob []byte,
) (*d3d12.ID3D12PipelineState, error) {
	if len(cachedBlob) > 0 {
		psoDesc.CachedPSO = d3d12.D3D12_CACHED_PIPELINE_STATE{
			CachedBlob:            unsafe.Pointer(&cachedBlob[0]),
			CachedBlobSizeInBytes: uintptr(len(cachedBlob)),
		}
		pso, err := d.raw.CreateGraphicsPipelineState(psoDesc)
		runtime.KeepAlive(cachedBlob)
		psoDesc.CachedPSO = d3d12.D3D12_CACHED_PIPELINE_STATE{}
		if err == nil {
			return pso, nil
		}
		if isInvalidCachedPSOError(err) {
			_ = d.psoCache.Delete(cacheKey)
		} else {
			return nil, err
		}
	}
	pso, err := d.raw.CreateGraphicsPipelineState(psoDesc)
	if err != nil {
		return nil, err
	}
	if d.psoCache != nil && cacheKey != "" {
		if blob, blobErr := pso.GetCachedBlob(); blobErr == nil {
			_ = d.psoCache.Save(cacheKey, blob)
		}
	}
	return pso, nil
}

// createComputePSO creates a compute PSO, loading/saving cached driver blobs.
func (d *Device) createComputePSO(
	psoDesc *d3d12.D3D12_COMPUTE_PIPELINE_STATE_DESC,
	cacheKey string,
	cachedBlob []byte,
) (*d3d12.ID3D12PipelineState, error) {
	if len(cachedBlob) > 0 {
		psoDesc.CachedPSO = d3d12.D3D12_CACHED_PIPELINE_STATE{
			CachedBlob:            unsafe.Pointer(&cachedBlob[0]),
			CachedBlobSizeInBytes: uintptr(len(cachedBlob)),
		}
		pso, err := d.raw.CreateComputePipelineState(psoDesc)
		runtime.KeepAlive(cachedBlob)
		psoDesc.CachedPSO = d3d12.D3D12_CACHED_PIPELINE_STATE{}
		if err == nil {
			return pso, nil
		}
		if isInvalidCachedPSOError(err) {
			_ = d.psoCache.Delete(cacheKey)
		} else {
			return nil, err
		}
	}
	pso, err := d.raw.CreateComputePipelineState(psoDesc)
	if err != nil {
		return nil, err
	}
	if d.psoCache != nil && cacheKey != "" {
		if blob, blobErr := pso.GetCachedBlob(); blobErr == nil {
			_ = d.psoCache.Save(cacheKey, blob)
		}
	}
	return pso, nil
}

// isInvalidCachedPSOError reports a PSO creation refusing its cached blob:
// one of another adapter or driver version, or a damaged one. Dawn retries
// without the blob on a driver version mismatch too.
func isInvalidCachedPSOError(err error) bool {
	if err == nil {
		return false
	}
	var hr d3d12.HRESULTError
	if errors.As(err, &hr) {
		switch hr {
		case d3d12.E_INVALIDARG, d3d12.D3D12_ERROR_DRIVER_VERSION_MISMATCH, d3d12.D3D12_ERROR_ADAPTER_NOT_FOUND:
			return true
		}
	}
	return false
}

// psoCacheKey names the disk cache directory of an adapter: its PCI ids
// and the version of its user-mode driver, whose blobs another driver
// version refuses. The LUID is left out, it holds only until a reboot.
func psoCacheKey(desc *dxgi.DXGI_ADAPTER_DESC1, driverVersion int64) string {
	var buf [24]byte
	binary.LittleEndian.PutUint32(buf[0:], desc.VendorID)
	binary.LittleEndian.PutUint32(buf[4:], desc.DeviceID)
	binary.LittleEndian.PutUint32(buf[8:], desc.SubSysID)
	binary.LittleEndian.PutUint32(buf[12:], desc.Revision)
	binary.LittleEndian.PutUint64(buf[16:], uint64(driverVersion))
	return pipelinecache.HexKey(buf[:])
}

// NewPSOBlobStore creates a store at UserCacheDir()/gogpu/dx12/<adapterKey>/.
func NewPSOBlobStore(adapterKey string) (*PSOBlobStore, error) {
	dir, err := pipelinecache.UserCachePath("dx12", adapterKey, "")
	if err != nil {
		return nil, err
	}
	return &PSOBlobStore{dir: dir}, nil
}

func (s *PSOBlobStore) blobPath(key string) string {
	return filepath.Join(s.dir, key+".pso")
}

// Load returns a cached PSO blob. Missing files return (nil, nil).
func (s *PSOBlobStore) Load(key string) ([]byte, error) {
	if s == nil || key == "" {
		return nil, nil
	}
	return pipelinecache.LoadBlob(s.blobPath(key))
}

// Save stores a PSO blob atomically.
func (s *PSOBlobStore) Save(key string, blob []byte) error {
	if s == nil || key == "" || len(blob) == 0 {
		return nil
	}
	return pipelinecache.SaveBlob(s.blobPath(key), blob)
}

// Delete removes a stale PSO blob.
func (s *PSOBlobStore) Delete(key string) error {
	if s == nil || key == "" {
		return nil
	}
	return pipelinecache.DeleteBlob(s.blobPath(key))
}
