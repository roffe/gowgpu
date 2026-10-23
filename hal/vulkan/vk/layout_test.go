//go:build !(js && wasm)

package vk_test

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/build"
	"go/importer"
	"go/parser"
	"go/token"
	"go/types"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
	"unicode"
)

// layoutKnownWrong are the structs whose Go layout is wrong on every target,
// as vk-gen writes them: C bitfields, and arrays vk-gen sizes as one element.
// gowgpu uses none of them.
var layoutKnownWrong = map[string]bool{
	"VkAccelerationStructureInstanceKHR":                               true,
	"VkAccelerationStructureMatrixMotionInstanceNV":                    true,
	"VkAccelerationStructureSRTMotionInstanceNV":                       true,
	"VkClusterAccelerationStructureBuildTriangleClusterInfoNV":         true,
	"VkClusterAccelerationStructureBuildTriangleClusterTemplateInfoNV": true,
	"VkClusterAccelerationStructureGeometryIndexAndGeometryFlagsNV":    true,
	"VkClusterAccelerationStructureInstantiateClusterInfoNV":           true,
	"VkPartitionedAccelerationStructureWriteInstanceDataNV":            true,
	"VkPhysicalDeviceDataGraphOperationSupportARM":                     true,
	"VkPipelineBinaryKeyKHR":                                           true,
	"VkPipelineCacheHeaderVersionDataGraphQCOM":                        true,
	"VkQueueFamilyDataGraphPropertiesARM":                              true,
	"VkQueueFamilyGlobalPriorityProperties":                            true,
	"VkTransformMatrixKHR":                                             true,
	"VkVideoDecodeVP9PictureInfoKHR":                                   true,
}

// TestStructLayoutsMatchC compares every struct of the bindings with its C
// declaration in vulkan_core.h, on 64-bit targets and on 32-bit ones, where
// non-dispatchable handles are 64 bits wide and arm and Windows on 386 align
// 64-bit members to 8 bytes: the size of each and the offset of each field,
// through static asserts clang checks for the target. It needs clang and the
// Vulkan headers, and skips without them.
func TestStructLayoutsMatchC(t *testing.T) {
	clang, err := exec.LookPath("clang")
	if err != nil {
		t.Skip("no clang")
	}
	inc := t.TempDir()
	for _, dir := range []string{"vulkan", "vk_video"} {
		if _, err := os.Stat(filepath.Join("/usr/include", dir)); err != nil {
			t.Skipf("no Vulkan headers: %v", err)
		}
		if err := os.Symlink(filepath.Join("/usr/include", dir), filepath.Join(inc, dir)); err != nil {
			t.Fatal(err)
		}
	}
	failed := regexp.MustCompile(`static assertion failed due to requirement .*: (size|offset) (Vk[A-Za-z0-9_]+)\b`)
	for _, target := range []struct{ goos, goarch, triple string }{
		{"linux", "amd64", "x86_64-linux-gnu"},
		{"linux", "arm64", "aarch64-linux-gnu"},
		{"linux", "386", "i686-linux-gnu"},
		{"linux", "arm", "armv7-linux-gnueabihf"},
		{"windows", "386", "i686-w64-windows-gnu"},
	} {
		t.Run(target.goos+"/"+target.goarch, func(t *testing.T) {
			src := filepath.Join(t.TempDir(), "layout.c")
			if err := os.WriteFile(src, layoutAsserts(t, target.goos, target.goarch), 0o644); err != nil {
				t.Fatal(err)
			}
			out, _ := exec.Command(clang, "--target="+target.triple, "-ffreestanding", "-fsyntax-only", "-ferror-limit=0", "-I", inc, src).CombinedOutput()
			wrong := map[string]bool{}
			for _, line := range strings.Split(string(out), "\n") {
				if m := failed.FindStringSubmatch(line); m != nil && !layoutKnownWrong[m[2]] && !wrong[m[2]] {
					wrong[m[2]] = true
					t.Errorf("%s: Go and C disagree: %s", m[2], line)
				}
			}
		})
	}
}

// layoutAsserts returns C static asserts that the structs of the package
// have the size and field offsets Go gives them on goos/goarch.
func layoutAsserts(t *testing.T, goos, goarch string) []byte {
	t.Helper()
	ctx := build.Default
	ctx.GOOS, ctx.GOARCH, ctx.CgoEnabled = goos, goarch, true
	bp, err := ctx.ImportDir(".", 0)
	if err != nil {
		t.Fatal(err)
	}
	fset := token.NewFileSet()
	var files []*ast.File
	for _, name := range append(bp.GoFiles, bp.CgoFiles...) {
		f, err := parser.ParseFile(fset, name, nil, 0)
		if err != nil {
			t.Fatal(err)
		}
		files = append(files, f)
	}
	sizes := types.SizesFor("gc", goarch)
	conf := types.Config{Sizes: sizes, FakeImportC: true, Importer: importer.Default(), Error: func(error) {}}
	pkg, _ := conf.Check("vk", fset, files, nil)
	var c bytes.Buffer
	c.WriteString("#include <stddef.h>\n#include <vulkan/vulkan_core.h>\n")
	for _, name := range pkg.Scope().Names() {
		tn, ok := pkg.Scope().Lookup(name).(*types.TypeName)
		if !ok || tn.IsAlias() {
			continue
		}
		st, ok := tn.Type().Underlying().(*types.Struct)
		if !ok || st.NumFields() == 0 {
			continue
		}
		fields := make([]*types.Var, st.NumFields())
		for i := range fields {
			fields[i] = st.Field(i)
		}
		offsets := sizes.Offsetsof(fields)
		cname := "Vk" + name
		fmt.Fprintf(&c, "_Static_assert(sizeof(%s) == %d, \"size %s\");\n", cname, sizes.Sizeof(tn.Type()), cname)
		for i, f := range fields {
			if f.Name() == "_" {
				continue
			}
			member := []rune(f.Name())
			member[0] = unicode.ToLower(member[0])
			fmt.Fprintf(&c, "_Static_assert(offsetof(%s, %s) == %d, \"offset %s.%s\");\n", cname, string(member), offsets[i], cname, string(member))
		}
	}
	return c.Bytes()
}
