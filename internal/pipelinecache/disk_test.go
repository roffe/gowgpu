// Copyright 2025 The GoGPU Authors
// SPDX-License-Identifier: MIT

package pipelinecache

import (
	"bytes"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
)

func TestUserCachePath(t *testing.T) {
	path, err := UserCachePath("vulkan", "abc123", "pipeline.cache")
	if err != nil {
		t.Fatal(err)
	}
	if !filepath.IsAbs(path) {
		t.Fatalf("expected absolute path, got %q", path)
	}
	if filepath.Base(path) != "pipeline.cache" {
		t.Fatalf("unexpected file name: %q", path)
	}
}

func TestUserCachePathError(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("Windows resolves UserCacheDir without HOME/USERPROFILE")
	}
	t.Setenv("HOME", "")
	t.Setenv("USERPROFILE", "")
	t.Setenv("XDG_CACHE_HOME", "")
	if _, err := UserCachePath("vulkan", "k", "f"); err == nil {
		t.Fatal("expected UserCachePath error with empty home/cache env")
	}
}

func TestSaveLoadDeleteBlob(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "nested", "cache.bin")

	data, err := LoadBlob(path)
	if err != nil {
		t.Fatal(err)
	}
	if data != nil {
		t.Fatalf("expected nil for missing file, got %d bytes", len(data))
	}

	payload := []byte("driver-isa-cache")
	if err := SaveBlob(path, payload); err != nil {
		t.Fatal(err)
	}

	loaded, err := LoadBlob(path)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(loaded, payload) {
		t.Fatalf("round-trip mismatch: %q vs %q", loaded, payload)
	}

	if err := DeleteBlob(path); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatalf("expected file removed, stat err=%v", err)
	}
}

func TestSaveBlobEmpty(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "empty.bin")
	if err := SaveBlob(path, nil); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatal("empty save should be a no-op")
	}
}

func TestLoadBlobNotAFile(t *testing.T) {
	dir := t.TempDir()
	if _, err := LoadBlob(dir); err == nil {
		t.Fatal("expected error reading a directory as blob")
	}
}

func TestSaveBlobMkdirFails(t *testing.T) {
	dir := t.TempDir()
	blocker := filepath.Join(dir, "not-a-dir")
	if err := os.WriteFile(blocker, []byte("x"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := SaveBlob(filepath.Join(blocker, "cache.bin"), []byte("data")); err == nil {
		t.Fatal("expected mkdir failure when parent is a file")
	}
}

func TestSaveBlobWriteFails(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("chmod semantics differ on Windows")
	}
	dir := t.TempDir()
	ro := filepath.Join(dir, "ro")
	if err := os.MkdirAll(ro, 0o750); err != nil {
		t.Fatal(err)
	}
	if err := os.Chmod(ro, 0o555); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.Chmod(ro, 0o750) })

	if err := SaveBlob(filepath.Join(ro, "cache.bin"), []byte("data")); err == nil {
		t.Fatal("expected write failure in read-only directory")
	}
}

func TestSaveBlobRenameFails(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "cache.bin")
	// Destination already exists as a non-empty directory → rename fails.
	if err := os.MkdirAll(filepath.Join(path, "child"), 0o750); err != nil {
		t.Fatal(err)
	}
	if err := SaveBlob(path, []byte("data")); err == nil {
		t.Fatal("expected rename failure when destination is a directory")
	}
}

func TestDeleteBlobMissing(t *testing.T) {
	if err := DeleteBlob(filepath.Join(t.TempDir(), "missing.bin")); err != nil {
		t.Fatal(err)
	}
}

func TestDeleteBlobError(t *testing.T) {
	dir := t.TempDir()
	nested := filepath.Join(dir, "subdir")
	if err := os.Mkdir(nested, 0o750); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(nested, "f"), []byte("x"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := DeleteBlob(nested); err == nil {
		t.Fatal("expected error deleting non-empty directory")
	}
}

func TestHexKeySkipsEmptyParts(t *testing.T) {
	a := HexKey([]byte("x"))
	b := HexKey(nil, []byte{}, []byte("x"))
	if a != b {
		t.Fatalf("empty parts should be skipped: %q vs %q", a, b)
	}
}

func TestAdapterKeysDeterministic(t *testing.T) {
	uuid := [16]byte{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
	k1 := VulkanAdapterKey(0x10DE, 0x2204, 999, uuid)
	k2 := VulkanAdapterKey(0x10DE, 0x2204, 999, uuid)
	if k1 != k2 {
		t.Fatalf("VulkanAdapterKey not deterministic: %q vs %q", k1, k2)
	}
	if VulkanAdapterKey(0x10DE, 0x2204, 998, uuid) == k1 {
		t.Fatal("driver version should affect VulkanAdapterKey")
	}
}

// TestLoadBlobRejectsPartialBlobs saves a blob and loads it back cut short,
// with a byte changed, and from a file SaveBlob did not write.
func TestLoadBlobRejectsPartialBlobs(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cache.bin")
	if err := SaveBlob(path, []byte("driver-isa-cache")); err != nil {
		t.Fatal(err)
	}
	whole, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	changed := bytes.Clone(whole)
	changed[len(changed)-1] ^= 1
	for name, file := range map[string][]byte{
		"cut short":    whole[:len(whole)-3],
		"byte changed": changed,
		"foreign":      []byte("driver-isa-cache"),
	} {
		if err := os.WriteFile(path, file, 0o600); err != nil {
			t.Fatal(err)
		}
		if data, err := LoadBlob(path); err == nil {
			t.Errorf("%s: LoadBlob returned %q and no error", name, data)
		}
	}
}

// TestConcurrentSavesLeaveOneWholeBlob saves different blobs to one path
// from several goroutines at once, as processes sharing an adapter's cache
// do: what is left is one of them, whole. Windows refuses a rename over a
// file another rename is replacing, so there some of the saves fail.
func TestConcurrentSavesLeaveOneWholeBlob(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cache.bin")
	var wg sync.WaitGroup
	var saved atomic.Int32
	for g := range 8 {
		wg.Go(func() {
			blob := bytes.Repeat([]byte{byte('a' + g)}, 64<<10+g)
			for range 20 {
				if err := SaveBlob(path, blob); err == nil {
					saved.Add(1)
				} else if runtime.GOOS != "windows" {
					t.Error(err)
					return
				}
			}
		})
	}
	wg.Wait()
	if saved.Load() == 0 {
		t.Fatal("no save succeeded")
	}
	data, err := LoadBlob(path)
	if err != nil {
		t.Fatal(err)
	}
	g := int(data[0] - 'a')
	if g < 0 || g >= 8 || !bytes.Equal(data, bytes.Repeat([]byte{data[0]}, 64<<10+g)) {
		t.Fatalf("left a blob of %d bytes starting %q, not one of those saved", len(data), data[:min(len(data), 8)])
	}
	if leftovers, _ := filepath.Glob(path + ".*.tmp"); len(leftovers) != 0 {
		t.Errorf("temporary files left behind: %v", leftovers)
	}
}
