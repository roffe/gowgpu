package window

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework QuartzCore
#import <AppKit/AppKit.h>
#import <QuartzCore/CAMetalLayer.h>

// metalLayer gives view a CAMetalLayer for Metal to present to.
static void *metalLayer(void *view) {
	NSView *v = (NSView *)view;
	CAMetalLayer *layer = [CAMetalLayer layer];
	layer.contentsScale = v.window.backingScaleFactor;
	[v setLayer:layer];
	[v setWantsLayer:YES];
	return layer;
}
*/
import "C"

import (
	"github.com/go-gl/glfw/v3.4/glfw"
	wgpu "github.com/roffe/gowgpu"
)

func surfaceTarget(win *glfw.Window) wgpu.SurfaceTargetUnsafe {
	return wgpu.SurfaceTargetFromMetalLayer(uintptr(C.metalLayer(win.GetCocoaView())))
}
