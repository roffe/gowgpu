// cgocheck checks a cgo package for a GOOS/GOARCH this machine has no
// sysroot for: it type checks the Go side with a faked "C" package and
// writes each file's cgo preamble out as a C (or Objective-C) file, so that
// clang --target=... -ffreestanding can compile the C side on its own.
//
//	cgocheck -goos ios -goarch arm64 -out /tmp/pre <package dir>
package main

import (
	"flag"
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
	"strings"
)

func main() {
	goos := flag.String("goos", "", "target GOOS")
	goarch := flag.String("goarch", "", "target GOARCH")
	tags := flag.String("tags", "", "comma separated build tags")
	out := flag.String("out", "", "directory to write the preambles to")
	cc := flag.String("cc", "", "run the real cgo tool with this C compiler and type check its output instead of faking package C")
	cflags := flag.String("cflags", "-ffreestanding", "C flags for -cc (the #cgo CFLAGS of the package are not applied)")
	flag.Parse()
	dir := flag.Arg(0)

	build.Default.GOOS, build.Default.GOARCH, build.Default.CgoEnabled = *goos, *goarch, true
	if *tags != "" {
		build.Default.BuildTags = strings.Split(*tags, ",")
	}
	bp, err := build.Default.ImportDir(dir, 0)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fset := token.NewFileSet()
	var files []*ast.File
	names := append([]string{}, bp.GoFiles...)
	var gen []string
	if *cc != "" && len(bp.CgoFiles) > 0 {
		objdir, err := os.MkdirTemp("", "cgocheck")
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		defer os.RemoveAll(objdir)
		args := []string{"tool", "cgo", "-objdir", objdir, "-importpath", bp.ImportPath, "--"}
		args = append(args, strings.Fields(*cflags)...)
		args = append(args, bp.CgoFiles...)
		cmd := exec.Command("go", args...)
		cmd.Dir = dir
		cmd.Env = append(os.Environ(), "GOOS="+*goos, "GOARCH="+*goarch, "CGO_ENABLED=1", "CC="+*cc)
		if b, err := cmd.CombinedOutput(); err != nil {
			fmt.Fprintf(os.Stderr, "go tool cgo: %v\n%s", err, b)
			os.Exit(1)
		}
		gen = append(gen, filepath.Join(objdir, "_cgo_gotypes.go"))
		for _, f := range bp.CgoFiles {
			gen = append(gen, filepath.Join(objdir, strings.TrimSuffix(f, ".go")+".cgo1.go"))
		}
		if *out != "" {
			for _, f := range bp.CgoFiles {
				pf, err := parser.ParseFile(fset, filepath.Join(dir, f), nil, parser.ParseComments)
				if err == nil {
					writePreamble(pf, f, *out)
				}
			}
		}
	} else {
		names = append(names, bp.CgoFiles...)
	}
	for _, g := range gen {
		f, err := parser.ParseFile(fset, g, nil, parser.ParseComments)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		files = append(files, f)
	}
	for _, name := range names {
		f, err := parser.ParseFile(fset, filepath.Join(dir, name), nil, parser.ParseComments)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		files = append(files, f)
		if *out != "" {
			writePreamble(f, name, *out)
		}
	}
	conf := types.Config{FakeImportC: *cc == "", Importer: importer.ForCompiler(fset, "source", nil)}
	nerr := 0
	conf.Error = func(err error) { nerr++; fmt.Println(err) }
	conf.Check(bp.ImportPath, fset, files, nil)
	mode := "fake C"
	if *cc != "" {
		mode = "real cgo output"
	}
	fmt.Printf("%s %s/%s (%s): %d Go files, %d cgo files, %d type errors\n", dir, *goos, *goarch, mode, len(bp.GoFiles), len(bp.CgoFiles), nerr)
	if nerr > 0 {
		os.Exit(1)
	}
}

// writePreamble saves the comment right before import "C", minus #cgo lines.
func writePreamble(f *ast.File, name, out string) {
	for _, imp := range f.Imports {
		if imp.Path.Value != `"C"` {
			continue
		}
		doc := imp.Doc
		if doc == nil {
			for _, d := range f.Decls {
				if g, ok := d.(*ast.GenDecl); ok && g.Tok == token.IMPORT && len(g.Specs) == 1 && g.Specs[0] == imp {
					doc = g.Doc
				}
			}
		}
		if doc == nil {
			return
		}
		var b strings.Builder
		ext := ".c"
		for _, line := range strings.Split(doc.Text(), "\n") {
			if strings.HasPrefix(strings.TrimSpace(line), "#cgo ") {
				if strings.Contains(line, "objective-c") {
					ext = ".m"
				}
				continue
			}
			b.WriteString(line + "\n")
		}
		os.MkdirAll(out, 0o755)
		os.WriteFile(filepath.Join(out, strings.TrimSuffix(name, ".go")+ext), []byte(b.String()), 0o644)
	}
}
