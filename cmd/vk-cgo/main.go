// Command vk-cgo rewrites the goffi Vulkan wrappers vk-gen writes
// (vk/commands_gen.go and vk/commands_manual.go) into cgo: every wrapper calls
// one C trampoline per distinct C signature, through the function pointer the
// loader resolved. Pointer arguments cross as const uint8_t *: a pointee cgo
// sees no pointers in, so cgocheck never walks the Go memory a create info
// points at, while the argument stays a real pointer that stack copying
// adjusts. Every trampoline is marked noescape and nocallback, so stack
// pointees stay on the stack and nothing allocates.
//
// It is the second half of regenerating the bindings, after vk-gen:
//
//	(cd cmd/vk-gen && go run . -spec vk.xml -out ../../hal/vulkan/vk/)
//	go run ./cmd/vk-cgo hal/vulkan/vk
//
// It rewrites both files in place, leaving wrappers already converted as they
// are, and writes trampolines.h next to them.
package main

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/format"
	"go/parser"
	"go/printer"
	"go/token"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

var typedefs = map[string]string{}

func collectTypedefs(f *ast.File) {
	for _, d := range f.Decls {
		g, ok := d.(*ast.GenDecl)
		if !ok || g.Tok != token.TYPE {
			continue
		}
		for _, s := range g.Specs {
			ts := s.(*ast.TypeSpec)
			switch t := ts.Type.(type) {
			case *ast.Ident:
				typedefs[ts.Name.Name] = t.Name
			case *ast.StarExpr, *ast.FuncType, *ast.SelectorExpr:
				typedefs[ts.Name.Name] = "*"
			case *ast.StructType:
				typedefs[ts.Name.Name] = "struct"
			case *ast.ArrayType:
				typedefs[ts.Name.Name] = "array"
			}
		}
	}
}

func resolve(name string) string {
	for range 10 {
		u, ok := typedefs[name]
		if !ok {
			break
		}
		name = u
	}
	return name
}

// class is the C ABI class of a Go type: h handle, p pointer, q 64-bit int,
// i 32-bit int, f float, d double.
func class(e ast.Expr) (string, error) {
	switch t := e.(type) {
	case *ast.StarExpr:
		return "p", nil
	case *ast.ArrayType: // a C array parameter: passed as a pointer to it
		return "a", nil
	case *ast.SelectorExpr:
		return "P", nil // unsafe.Pointer
	case *ast.Ident:
		switch b := resolve(t.Name); b {
		case "uintptr":
			return "h", nil
		case "*":
			return "P", nil
		case "uint64", "int64":
			return "q", nil
		case "uint32", "int32":
			return "i", nil
		case "float32":
			return "f", nil
		case "float64":
			return "d", nil
		default:
			return "", fmt.Errorf("unsupported type %s (%s)", t.Name, b)
		}
	}
	return "", fmt.Errorf("unsupported type expression %T", e)
}

var ctype = map[byte]string{'h': "uint64_t", 'p': "const uint8_t *", 'P': "const uint8_t *", 'a': "const uint8_t *", 'q': "uint64_t", 'i': "uint32_t", 'f': "float", 'd': "double", 'v': "void"}

func trampolineName(ret string, params []string) string {
	s := strings.Join(params, "")
	s = strings.NewReplacer("P", "p", "a", "p").Replace(s)
	return "gvk_" + ret + "_" + s
}

type param struct {
	name, cls string
	typ       ast.Expr
}

var trampolines = map[string]string{} // name -> C definition

var fileUses map[string]bool // trampolines the file being rewritten calls

func addTrampoline(name, ret string, params []param) {
	if _, ok := trampolines[name]; ok {
		return
	}
	var proto, args, call []string
	for i, p := range params {
		t := ctype[p.cls[0]]
		pt := t
		if strings.HasSuffix(t, "*") {
			pt = "const void *"
		}
		proto = append(proto, pt)
		args = append(args, fmt.Sprintf("%s a%d", t, i))
		call = append(call, fmt.Sprintf("a%d", i))
	}
	rt := ctype[ret[0]]
	if ret == "r" {
		rt = "int32_t"
	}
	body := fmt.Sprintf("((gvk_pfn_%s)fn)(%s);", strings.TrimPrefix(name, "gvk_"), strings.Join(call, ", "))
	if rt != "void" {
		body = "return " + body
	}
	trampolines[name] = fmt.Sprintf("typedef %s (VKAPI_PTR *gvk_pfn_%s)(%s);\nstatic inline %s %s(uintptr_t fn, %s) { %s }\n",
		rt, strings.TrimPrefix(name, "gvk_"), strings.Join(proto, ", "), rt, name, strings.Join(args, ", "), body)
}

// keepTrampoline registers the trampoline an already converted wrapper calls,
// from its name: gvk_<result class>_<parameter classes>.
func keepTrampoline(name string) {
	ret, cls, _ := strings.Cut(strings.TrimPrefix(name, "gvk_"), "_")
	params := make([]param, len(cls))
	for i := range cls {
		params[i].cls = cls[i : i+1]
	}
	addTrampoline(name, ret, params)
	fileUses[name] = true
}

func exprString(fset *token.FileSet, e ast.Node) string {
	var b bytes.Buffer
	printer.Fprint(&b, fset, e)
	return b.String()
}

// convert rewrites one method; ok is false when it has no goffi call.
func convert(fset *token.FileSet, fd *ast.FuncDecl) (string, bool, error) {
	var field string
	ast.Inspect(fd.Body, func(n ast.Node) bool {
		c, ok := n.(*ast.CallExpr)
		if !ok {
			return true
		}
		if s, ok := c.Fun.(*ast.SelectorExpr); ok && s.Sel.Name == "CallFunction" {
			if f, ok := c.Args[1].(*ast.SelectorExpr); ok {
				field = f.Sel.Name
			}
		}
		return true
	})
	if field == "" {
		// Already converted: the method stays as it is, and so does the
		// trampoline it calls, which its name describes.
		ast.Inspect(fd.Body, func(n ast.Node) bool {
			if s, ok := n.(*ast.SelectorExpr); ok {
				if x, ok := s.X.(*ast.Ident); ok && x.Name == "C" && strings.HasPrefix(s.Sel.Name, "gvk_") {
					keepTrampoline(s.Sel.Name)
				}
			}
			return true
		})
		return "", false, nil
	}
	var params []param
	for _, p := range fd.Type.Params.List {
		cls, err := class(p.Type)
		if err != nil {
			return "", false, fmt.Errorf("%s: %v", fd.Name.Name, err)
		}
		for _, n := range p.Names {
			params = append(params, param{n.Name, cls, p.Type})
		}
	}
	ret, retType := "v", ""
	if fd.Type.Results != nil {
		retType = exprString(fset, fd.Type.Results.List[0].Type)
		c, err := class(fd.Type.Results.List[0].Type)
		if err != nil {
			return "", false, fmt.Errorf("%s result: %v", fd.Name.Name, err)
		}
		ret = c
		if retType == "Result" {
			ret = "r"
		}
	}
	var cls []string
	for _, p := range params {
		cls = append(cls, p.cls)
	}
	name := trampolineName(ret, cls)
	addTrampoline(name, ret, params)
	fileUses[name] = true

	var b strings.Builder
	if fd.Doc != nil {
		for _, c := range fd.Doc.List {
			b.WriteString(c.Text + "\n")
		}
	}
	b.WriteString("func (c *Commands) " + fd.Name.Name + exprString(fset, fd.Type)[len("func"):] + " {\n")
	zero := ""
	switch {
	case ret == "r":
		zero = " ErrorInitializationFailed"
	case ret != "v":
		zero = " 0"
	}
	fmt.Fprintf(&b, "\tif c.%s == nil {\n\t\treturn%s\n\t}\n", field, zero)
	var args []string
	args = append(args, fmt.Sprintf("C.uintptr_t(uintptr(c.%s))", field))
	var keep []string
	for _, p := range params {
		switch p.cls {
		case "h", "q":
			args = append(args, fmt.Sprintf("C.uint64_t(%s)", p.name))
		case "i":
			args = append(args, fmt.Sprintf("C.uint32_t(%s)", p.name))
		case "f":
			args = append(args, fmt.Sprintf("C.float(%s)", p.name))
		case "d":
			args = append(args, fmt.Sprintf("C.double(%s)", p.name))
		case "p":
			args = append(args, fmt.Sprintf("(*C.uint8_t)(unsafe.Pointer(%s))", p.name))
		case "P":
			args = append(args, fmt.Sprintf("(*C.uint8_t)(%s)", p.name))
		case "a":
			args = append(args, fmt.Sprintf("(*C.uint8_t)(unsafe.Pointer(&%s))", p.name))
		}
	}
	call := fmt.Sprintf("C.%s(%s)", name, strings.Join(args, ", "))
	if ret == "v" {
		b.WriteString("\t" + call + "\n")
	} else {
		b.WriteString("\tr := " + call + "\n")
	}
	for _, k := range keep {
		fmt.Fprintf(&b, "\truntime.KeepAlive(%s)\n", k)
	}
	if ret != "v" {
		fmt.Fprintf(&b, "\treturn %s(r)\n", retType)
	}
	b.WriteString("}\n")
	return b.String(), true, nil
}

func rewrite(dir, name string) (int, error) {
	path := filepath.Join(dir, name)
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, path, nil, parser.ParseComments)
	if err != nil {
		return 0, err
	}
	fileUses = map[string]bool{}
	var body bytes.Buffer
	var out bytes.Buffer
	// Keep the build constraint and the leading comments.
	for _, cg := range f.Comments {
		if cg.Pos() >= f.Package {
			break
		}
		for _, c := range cg.List {
			out.WriteString(c.Text + "\n")
		}
		out.WriteString("\n")
	}
	n := 0
	for _, d := range f.Decls {
		if g, ok := d.(*ast.GenDecl); ok && g.Tok == token.IMPORT {
			continue
		}
		if g, ok := d.(*ast.GenDecl); ok && g.Tok == token.VAR {
			// drop "var _ = unsafe.Sizeof(0)"
			if len(g.Specs) == 1 {
				if vs, ok := g.Specs[0].(*ast.ValueSpec); ok && len(vs.Names) == 1 && vs.Names[0].Name == "_" {
					continue
				}
			}
		}
		if fd, ok := d.(*ast.FuncDecl); ok && fd.Recv != nil {
			src, ok, err := convert(fset, fd)
			if err != nil {
				return 0, err
			}
			if ok {
				body.WriteString(src + "\n")
				n++
				continue
			}
		}
		var b bytes.Buffer
		printer.Fprint(&b, fset, &printer.CommentedNode{Node: d, Comments: f.Comments})
		body.Write(b.Bytes())
		body.WriteString("\n\n")
	}
	used := make([]string, 0, len(fileUses))
	for u := range fileUses {
		used = append(used, u)
	}
	sort.Strings(used)
	out.WriteString("package vk\n\n/*\n#include \"trampolines.h\"\n")
	for _, u := range used {
		fmt.Fprintf(&out, "#cgo noescape %s\n#cgo nocallback %s\n", u, u)
	}
	out.WriteString("*/\nimport \"C\"\n\nimport \"unsafe\"\n\nvar _ = unsafe.Pointer(nil)\n\n")
	out.Write(body.Bytes())
	src, err := format.Source(out.Bytes())
	if err != nil {
		os.WriteFile(path+".broken", out.Bytes(), 0o644)
		return 0, fmt.Errorf("%s: %v", name, err)
	}
	return n, os.WriteFile(path, src, 0o644)
}

func main() {
	dir := os.Args[1]
	ents, _ := os.ReadDir(dir)
	fset := token.NewFileSet()
	for _, e := range ents {
		if strings.HasSuffix(e.Name(), ".go") && !strings.HasSuffix(e.Name(), "_test.go") {
			f, err := parser.ParseFile(fset, filepath.Join(dir, e.Name()), nil, 0)
			if err != nil {
				panic(err)
			}
			collectTypedefs(f)
		}
	}
	total := 0
	for _, name := range []string{"commands_gen.go", "commands_manual.go"} {
		n, err := rewrite(dir, name)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		fmt.Printf("%s: %d wrappers\n", name, n)
		total += n
	}
	names := make([]string, 0, len(trampolines))
	for n := range trampolines {
		names = append(names, n)
	}
	sort.Strings(names)
	var h bytes.Buffer
	h.WriteString(`// Code generated by vkcgo. DO NOT EDIT.

// One trampoline per C signature of the Vulkan commands gogpu wraps. No
// Vulkan headers: handles are uint64_t, pointers uintptr_t, and fn is the
// pointer vkGet*ProcAddr returned.
#ifndef GVK_TRAMPOLINES_H
#define GVK_TRAMPOLINES_H
#include <stdint.h>

#if defined(__arm__)
#define VKAPI_PTR __attribute__((pcs("aapcs-vfp")))
#elif defined(_WIN32) && !defined(_WIN64)
#define VKAPI_PTR __stdcall
#else
#define VKAPI_PTR
#endif

`)
	for _, n := range names {
		h.WriteString(trampolines[n])
	}
	h.WriteString("\n#endif\n")
	if err := os.WriteFile(filepath.Join(dir, "trampolines.h"), h.Bytes(), 0o644); err != nil {
		panic(err)
	}
	fmt.Printf("%d wrappers, %d trampolines\n", total, len(names))
}
