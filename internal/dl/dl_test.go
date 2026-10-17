//go:build linux

package dl

import "testing"

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
