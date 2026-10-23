//go:build linux

package dl

import (
	"fmt"
	"strings"
	"sync"
	"testing"
)

// libc is always there, and it has strlen but no symbol by this name.
func TestOpenAndSym(t *testing.T) {
	lib, err := Open("libdoesnotexist.so", "libc.so.6")
	if err != nil {
		t.Fatal(err)
	}
	if p, err := lib.Sym("strlen"); err != nil || p == nil {
		t.Fatalf("strlen: %v %v", p, err)
	}
	if _, err := lib.Sym("no_such_symbol_here"); err == nil {
		t.Fatal("missing symbol found")
	}
	if lib.SymOrNil("no_such_symbol_here") != nil {
		t.Fatal("SymOrNil found a missing symbol")
	}
	if _, err := Open("libdoesnotexist.so"); err == nil {
		t.Fatal("missing library opened")
	}
}

// dlerror's message is the calling thread's, so each failed Open and Sym
// reports its own one, also with others failing on other goroutines.
func TestErrorsAreTheirOwn(t *testing.T) {
	libc, err := Open("libc.so.6")
	if err != nil {
		t.Fatal(err)
	}
	var wg sync.WaitGroup
	errs := make(chan string, 64)
	for g := range 32 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range 500 {
				name := fmt.Sprintf("libnope-%d-%d.so", g, i)
				if _, err := Open(name); !aboutItself(err, name) {
					errs <- fmt.Sprintf("Open(%s): %v", name, err)
					return
				}
				sym := fmt.Sprintf("nope_%d_%d", g, i)
				if _, err := libc.Sym(sym); !aboutItself(err, sym) {
					errs <- fmt.Sprintf("Sym(%s): %v", sym, err)
					return
				}
			}
		}()
	}
	wg.Wait()
	close(errs)
	for e := range errs {
		t.Error(e)
	}
}

// aboutItself reports whether err, the error of a dl call about name, has
// the message dlerror gave for it, which names it again.
func aboutItself(err error, name string) bool {
	return err != nil && strings.Contains(strings.TrimPrefix(err.Error(), name+": "), name)
}
