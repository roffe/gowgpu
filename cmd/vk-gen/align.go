package main

import (
	"fmt"
	"go/ast"
	"go/build"
	"go/importer"
	"go/parser"
	"go/token"
	"go/types"
	"path/filepath"
	"strconv"
	"strings"
)

// align8Pads lays the structs of the bindings in outDir out as C does on a
// 32-bit target that aligns 64-bit members to 8 bytes (arm, mips, Windows on
// 386), where Go aligns them to 4. For each struct whose Go layout differs it
// returns the padding to put ahead of each field and, last, after the last
// field. Structs with bitfields, which have no Go counterpart, are left out.
func align8Pads(outDir string, bitfields map[string]bool) (map[string][]int64, error) {
	ctx := build.Default
	ctx.GOOS, ctx.GOARCH, ctx.CgoEnabled = "linux", "arm", true
	bp, err := ctx.ImportDir(outDir, 0)
	if err != nil {
		return nil, err
	}
	fset := token.NewFileSet()
	var files []*ast.File
	for _, name := range append(bp.GoFiles, bp.CgoFiles...) {
		f, err := parser.ParseFile(fset, filepath.Join(outDir, name), nil, 0)
		if err != nil {
			return nil, err
		}
		// The types, not the command wrappers, which call through cgo or,
		// before vk-cgo rewrites them, a package that is gone.
		if importsAny(f, "C", "github.com/go-webgpu/goffi/ffi") {
			continue
		}
		files = append(files, f)
	}
	conf := types.Config{Importer: importer.Default(), Error: func(error) {}}
	pkg, _ := conf.Check("vk", fset, files, nil)
	c := cLayout{ptr: 4}
	pads := map[string][]int64{}
	for _, name := range pkg.Scope().Names() {
		tn, ok := pkg.Scope().Lookup(name).(*types.TypeName)
		if !ok || tn.IsAlias() || bitfields[name] {
			continue
		}
		st, ok := tn.Type().Underlying().(*types.Struct)
		if !ok || st.NumFields() == 0 {
			continue
		}
		for i := range st.NumFields() {
			if !c.valid(st.Field(i).Type()) {
				return nil, fmt.Errorf("%s.%s: no layout for %s", name, st.Field(i).Name(), st.Field(i).Type())
			}
		}
		if pad := c.goPads(st); pad != nil {
			pads[name] = pad
		}
	}
	return pads, nil
}

// bitfieldStructs returns the Go names of the structs with bitfield members.
func bitfieldStructs(registry *Registry) map[string]bool {
	set := map[string]bool{}
	for _, t := range registry.Types.Types {
		for _, m := range t.Members {
			if strings.Contains(m.RawXML, "</name>:") {
				set[vkToGoType(t.Name)] = true
			}
		}
	}
	return set
}

func importsAny(f *ast.File, paths ...string) bool {
	for _, imp := range f.Imports {
		p, _ := strconv.Unquote(imp.Path.Value)
		for _, want := range paths {
			if p == want {
				return true
			}
		}
	}
	return false
}

// cLayout is C's layout of Go types on a target with ptr-byte pointers that
// aligns 64-bit members to 8 bytes.
type cLayout struct{ ptr int64 }

func (c cLayout) valid(t types.Type) bool {
	switch u := t.Underlying().(type) {
	case *types.Basic:
		return u.Kind() != types.Invalid && u.Info()&types.IsString == 0
	case *types.Pointer, *types.Signature:
		return true
	case *types.Array:
		return c.valid(u.Elem())
	case *types.Struct:
		for i := range u.NumFields() {
			if !c.valid(u.Field(i).Type()) {
				return false
			}
		}
		return true
	}
	return false
}

func (c cLayout) align(t types.Type) int64 {
	switch u := t.Underlying().(type) {
	case *types.Array:
		return c.align(u.Elem())
	case *types.Struct:
		a := int64(1)
		for i := range u.NumFields() {
			a = max(a, c.align(u.Field(i).Type()))
		}
		return a
	}
	return c.size(t)
}

func (c cLayout) size(t types.Type) int64 {
	switch u := t.Underlying().(type) {
	case *types.Basic:
		switch u.Kind() {
		case types.Int64, types.Uint64, types.Float64:
			return 8
		case types.Int32, types.Uint32, types.Float32:
			return 4
		case types.Int16, types.Uint16:
			return 2
		case types.Int8, types.Uint8, types.Bool:
			return 1
		}
		return c.ptr // uintptr, unsafe.Pointer
	case *types.Array:
		if u.Len() == 0 {
			return 0
		}
		return roundUp(c.size(u.Elem()), c.align(u.Elem())) * u.Len()
	case *types.Struct:
		n := u.NumFields()
		if n == 0 {
			return 0
		}
		off := c.offsets(u)
		return roundUp(off[n-1]+c.size(u.Field(n-1).Type()), c.align(u))
	}
	return c.ptr // pointers and functions
}

func (c cLayout) offsets(st *types.Struct) []int64 {
	off := make([]int64, st.NumFields())
	o := int64(0)
	for i := range st.NumFields() {
		ft := st.Field(i).Type()
		o = roundUp(o, c.align(ft))
		off[i] = o
		o += c.size(ft)
	}
	return off
}

// goPads returns the padding Go needs ahead of each field of st and after
// the last to lay it out as C does, where Go aligns nothing to more than 4
// bytes and a struct field is as large as C's, being padded itself; nil if
// it needs none.
func (c cLayout) goPads(st *types.Struct) []int64 {
	n := st.NumFields()
	pad := make([]int64, n+1)
	off := c.offsets(st)
	need := false
	o, goAlign := int64(0), int64(1)
	for i := range n {
		ft := st.Field(i).Type()
		a := min(c.align(ft), 4)
		goAlign = max(goAlign, a)
		o = roundUp(o, a)
		if off[i] > o {
			pad[i], o, need = off[i]-o, off[i], true
		}
		o += c.size(ft)
	}
	if size := c.size(st); size > roundUp(o, goAlign) {
		pad[n], need = size-roundUp(o, goAlign), true
	}
	if !need {
		return nil
	}
	return pad
}

func roundUp(x, a int64) int64 { return (x + a - 1) / a * a }
